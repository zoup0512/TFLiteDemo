package com.ns.tflite.demo

import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.ns.tflite.demo.ui.theme.TFLiteDemoTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    // 1. 声明 TFLite 解释器（全局变量，避免重复加载）
    private var tfliteInterpreter: Interpreter? = null
    // 存储推理结果的状态变量（用于 Compose UI 更新）
    private var inferenceResult by mutableStateOf("未加载模型")

    private var inputBitmap by mutableStateOf<Bitmap?>(null)
    private var outputBitmap by mutableStateOf<Bitmap?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // 2. 初始化 TFLite 模型（App 启动时加载）
        initTFLiteModel()

        // 3. 加载测试图片（assets/test.png）
        inputBitmap = loadBitmapFromAssets("test.png")

        setContent {
            TFLiteDemoTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    // 3. 重构 UI，新增模型推理按钮和结果展示
                    TFLiteDemoUI(
                        modifier = Modifier.padding(innerPadding),
                        inferenceResult = inferenceResult,
                        inputBitmap = inputBitmap,
                        outputBitmap = outputBitmap,
                        onInferenceClick = { runModelInference() }
                    )
                }
            }
        }

        // 启动后自动跑一次推理（如果图片/模型都准备好）
        runModelInference()
    }

    // 核心方法1：加载 assets 中的 TFLite 模型
    private fun initTFLiteModel() {
        try {
            // 替换为你的 .tflite 模型文件名（需放到 src/main/assets 目录下）
            val modelFile = loadModelFromAssets("model.tflite")
            tfliteInterpreter = Interpreter(modelFile)
            inferenceResult = "✅ 模型加载成功！"
        } catch (e: Exception) {
            inferenceResult = "❌ 模型加载失败：${e.message}"
            e.printStackTrace()
        }
    }

    // 核心方法2：从 assets 读取模型文件
    @Throws(Exception::class)
    private fun loadModelFromAssets(modelName: String): MappedByteBuffer {
        val assetFileDescriptor: AssetFileDescriptor = assets.openFd(modelName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadBitmapFromAssets(assetName: String): Bitmap? {
        return try {
            assets.open(assetName).use { input ->
                BitmapFactory.decodeStream(input)
            }
        } catch (e: Exception) {
            inferenceResult = "❌ 读取 assets/$assetName 失败：${e.message}"
            null
        }
    }

    // 核心方法3：运行模型推理
    private fun runModelInference() {
        val interpreter = tfliteInterpreter
        if (interpreter == null) {
            inferenceResult = "❌ 模型未加载，无法推理"
            return
        }
        val srcBitmap = inputBitmap
        if (srcBitmap == null) {
            inferenceResult = "❌ 输入图片未加载（assets/test.png）"
            return
        }

        inferenceResult = "⏳ 推理中..."
        lifecycleScope.launch {
            try {
                val outBmp = withContext(Dispatchers.Default) {
                    runImageToImage(interpreter, srcBitmap)
                }
                outputBitmap = outBmp
                inferenceResult = "✅ 推理成功"
            } catch (e: Exception) {
                inferenceResult = "❌ 推理失败：${e.message}"
                e.printStackTrace()
            }
        }
    }

    private enum class TensorLayout {
        NHWC,
        NCHW
    }

    private data class ImageTensorSpec(
        val layout: TensorLayout,
        val batch: Int,
        val height: Int,
        val width: Int,
        val channels: Int
    )

    private fun parseImageTensorSpec(shape: IntArray): ImageTensorSpec {
        if (shape.size != 4) {
            throw IllegalStateException("仅支持 4D 张量，实际为：${shape.contentToString()}")
        }

        val n = shape[0]
        val d1 = shape[1]
        val d2 = shape[2]
        val d3 = shape[3]

        // Heuristic:
        // - NHWC: [1, H, W, C] where C is 1 or 3
        // - NCHW: [1, C, H, W] where C is 1 or 3
        val nhwc = (d3 == 1 || d3 == 3)
        val nchw = (d1 == 1 || d1 == 3)

        return when {
            nhwc && !nchw -> ImageTensorSpec(TensorLayout.NHWC, n, d1, d2, d3)
            nchw && !nhwc -> ImageTensorSpec(TensorLayout.NCHW, n, d2, d3, d1)
            nhwc && nchw -> {
                // ambiguous (e.g. [1,3,3,3]) - prefer NHWC
                ImageTensorSpec(TensorLayout.NHWC, n, d1, d2, d3)
            }
            else -> {
                // fallback: assume NHWC
                ImageTensorSpec(TensorLayout.NHWC, n, d1, d2, d3)
            }
        }
    }

    private fun runImageToImage(interpreter: Interpreter, srcBitmap: Bitmap): Bitmap {
        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape() // e.g. [1,h,w,3]
        val inputType = inputTensor.dataType()

        val inputSpec = parseImageTensorSpec(inputShape)
        if (inputSpec.batch != 1) {
            throw IllegalStateException("仅支持 batch=1，实际为：${inputSpec.batch}")
        }
        if (inputSpec.channels != 3) {
            throw IllegalStateException("仅支持 3 通道 RGB 输入，实际通道数：${inputSpec.channels}")
        }

        val inH = inputSpec.height
        val inW = inputSpec.width

        val resized = if (srcBitmap.width != inW || srcBitmap.height != inH) {
            Bitmap.createScaledBitmap(srcBitmap, inW, inH, true)
        } else {
            srcBitmap
        }

        val inputBuffer = when (inputType) {
            org.tensorflow.lite.DataType.FLOAT32 -> bitmapToFloatBuffer(resized)
            org.tensorflow.lite.DataType.UINT8 -> bitmapToUInt8Buffer(resized)
            else -> throw IllegalStateException("不支持的输入 dtype：$inputType")
        }

        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape() // expect [1,h,w,3]
        val outputType = outputTensor.dataType()

        val outputSpec = parseImageTensorSpec(outputShape)
        if (outputSpec.batch != 1) {
            throw IllegalStateException("仅支持 batch=1 输出，实际为：${outputSpec.batch}")
        }
        if (outputSpec.channels != 1 && outputSpec.channels != 3) {
            throw IllegalStateException("仅支持 1/3 通道输出，实际通道数：${outputSpec.channels}")
        }

        val outH = outputSpec.height
        val outW = outputSpec.width
        val outC = outputSpec.channels

        inferenceResult = "input=${inputShape.contentToString()} $inputType (${inputSpec.layout}), output=${outputShape.contentToString()} $outputType (${outputSpec.layout})"

        val outputBuffer: ByteBuffer = ByteBuffer.allocateDirect(
            outH * outW * outC * when (outputType) {
                org.tensorflow.lite.DataType.FLOAT32 -> 4
                org.tensorflow.lite.DataType.UINT8 -> 1
                else -> throw IllegalStateException("不支持的输出 dtype：$outputType")
            }
        ).order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        return when (outputType) {
            org.tensorflow.lite.DataType.FLOAT32 -> floatBufferToBitmap(outputBuffer, outW, outH, outC, outputSpec.layout)
            org.tensorflow.lite.DataType.UINT8 -> uint8BufferToBitmap(outputBuffer, outW, outH, outC, outputSpec.layout)
            else -> throw IllegalStateException("不支持的输出 dtype：$outputType")
        }
    }

    private fun bitmapToUInt8Buffer(bitmap: Bitmap): ByteBuffer {
        val w = bitmap.width
        val h = bitmap.height
        val buffer = ByteBuffer.allocateDirect(w * h * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        for (i in pixels.indices) {
            val c = pixels[i]
            buffer.put(((c shr 16) and 0xFF).toByte())
            buffer.put(((c shr 8) and 0xFF).toByte())
            buffer.put((c and 0xFF).toByte())
        }
        buffer.rewind()
        return buffer
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): ByteBuffer {
        val w = bitmap.width
        val h = bitmap.height
        val buffer = ByteBuffer.allocateDirect(w * h * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        for (i in pixels.indices) {
            val c = pixels[i]
            buffer.putFloat(((c shr 16) and 0xFF) / 255f)
            buffer.putFloat(((c shr 8) and 0xFF) / 255f)
            buffer.putFloat((c and 0xFF) / 255f)
        }
        buffer.rewind()
        return buffer
    }

    private fun uint8BufferToBitmap(
        buffer: ByteBuffer,
        width: Int,
        height: Int,
        channels: Int,
        layout: TensorLayout
    ): Bitmap {
        val pixels = IntArray(width * height)
        when (layout) {
            TensorLayout.NHWC -> {
                for (i in pixels.indices) {
                    if (channels == 1) {
                        val v = buffer.get().toInt() and 0xFF
                        pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                    } else {
                        val r = buffer.get().toInt() and 0xFF
                        val g = buffer.get().toInt() and 0xFF
                        val b = buffer.get().toInt() and 0xFF
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
            }
            TensorLayout.NCHW -> {
                // buffer is [C][H][W]
                val planeSize = width * height
                val rPlane = ByteArray(planeSize)
                val gPlane = if (channels == 3) ByteArray(planeSize) else null
                val bPlane = if (channels == 3) ByteArray(planeSize) else null
                buffer.get(rPlane)
                if (channels == 3) {
                    buffer.get(gPlane!!)
                    buffer.get(bPlane!!)
                }
                for (i in pixels.indices) {
                    if (channels == 1) {
                        val v = rPlane[i].toInt() and 0xFF
                        pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                    } else {
                        val r = rPlane[i].toInt() and 0xFF
                        val g = gPlane!![i].toInt() and 0xFF
                        val b = bPlane!![i].toInt() and 0xFF
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun floatBufferToBitmap(
        buffer: ByteBuffer,
        width: Int,
        height: Int,
        channels: Int,
        layout: TensorLayout
    ): Bitmap {
        val pixels = IntArray(width * height)
        when (layout) {
            TensorLayout.NHWC -> {
                for (i in pixels.indices) {
                    if (channels == 1) {
                        val v = (buffer.getFloat() * 255f).toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                    } else {
                        val r = (buffer.getFloat() * 255f).toInt().coerceIn(0, 255)
                        val g = (buffer.getFloat() * 255f).toInt().coerceIn(0, 255)
                        val b = (buffer.getFloat() * 255f).toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
            }
            TensorLayout.NCHW -> {
                val planeSize = width * height
                val rPlane = FloatArray(planeSize)
                val gPlane = if (channels == 3) FloatArray(planeSize) else null
                val bPlane = if (channels == 3) FloatArray(planeSize) else null

                for (i in 0 until planeSize) rPlane[i] = buffer.getFloat()
                if (channels == 3) {
                    for (i in 0 until planeSize) gPlane!![i] = buffer.getFloat()
                    for (i in 0 until planeSize) bPlane!![i] = buffer.getFloat()
                }

                for (i in pixels.indices) {
                    if (channels == 1) {
                        val v = (rPlane[i] * 255f).toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                    } else {
                        val r = (rPlane[i] * 255f).toInt().coerceIn(0, 255)
                        val g = (gPlane!![i] * 255f).toInt().coerceIn(0, 255)
                        val b = (bPlane!![i] * 255f).toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    // 页面销毁时释放资源（避免内存泄漏）
    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter?.close()
        tfliteInterpreter = null
    }
}

// Compose UI 组件（抽离成独立函数，更易维护）
@Composable
fun TFLiteDemoUI(
    modifier: Modifier = Modifier,
    inferenceResult: String,
    inputBitmap: Bitmap?,
    outputBitmap: Bitmap?,
    onInferenceClick: () -> Unit
) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .padding(20.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Text(text = "TFLite 本地推理 Demo")

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(text = "输入图片")
                    Text(text = inputBitmap?.let { "${it.width} x ${it.height}" } ?: "-")
                    if (inputBitmap != null) {
                        Image(
                            bitmap = inputBitmap.asImageBitmap(),
                            contentDescription = "input",
                            modifier = Modifier
                                .padding(top = 8.dp)
                                .size(160.dp)
                        )
                    } else {
                        Text(text = "(未加载)")
                    }
                }

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(text = "输出图片")
                    Text(text = outputBitmap?.let { "${it.width} x ${it.height}" } ?: "-")
                    if (outputBitmap != null) {
                        Image(
                            bitmap = outputBitmap.asImageBitmap(),
                            contentDescription = "output",
                            modifier = Modifier
                                .padding(top = 8.dp)
                                .size(160.dp)
                        )
                    } else {
                        Text(text = "(暂无)")
                    }
                }
            }

            // 推理按钮
            Button(
                onClick = onInferenceClick,
                modifier = Modifier.padding(top = 20.dp, bottom = 20.dp)
            ) {
                Text(text = "运行模型推理")
            }

            // 推理结果展示
            Text(
                text = inferenceResult,
                modifier = Modifier.padding(top = 10.dp)
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun TFLiteDemoPreview() {
    TFLiteDemoTheme {
        TFLiteDemoUI(
            inferenceResult = "预览：模型未加载",
            inputBitmap = null,
            outputBitmap = null,
            onInferenceClick = {}
        )
    }
}