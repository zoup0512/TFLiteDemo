package com.ns.tflite.demo

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer

class MainActivity : ComponentActivity() {
    // 1. 替换为 ONNX Runtime 的 Session（全局变量）
    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null
    // 存储推理结果的状态变量
    private var inferenceResult by mutableStateOf("未加载模型")

    private var inputBitmap by mutableStateOf<Bitmap?>(null)
    private var outputBitmap by mutableStateOf<Bitmap?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // 初始化 ONNX 环境和模型
        initONNXModel()

        // 加载测试图片（assets/test.png）
        val loadedBitmap = loadBitmapFromAssets("test.png")
        inputBitmap = loadedBitmap
        if (loadedBitmap == null) {
            inferenceResult = "❌ 请在 assets 目录添加 test.png 图片"
        } else {
            android.util.Log.d("ONNX_DEBUG", "输入图片: ${loadedBitmap.width}x${loadedBitmap.height}")
        }

        setContent {
            TFLiteDemoTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    ONNXDemoUI(
                        modifier = Modifier.padding(innerPadding),
                        inferenceResult = inferenceResult,
                        inputBitmap = inputBitmap,
                        outputBitmap = outputBitmap,
                        onInferenceClick = { runModelInference() }
                    )
                }
            }
        }

        // 启动后自动跑一次推理
        runModelInference()
    }

    // 核心方法1：初始化 ONNX 模型（替换原 TFLite 初始化）
    private fun initONNXModel() {
        try {
            // 初始化 ONNX 环境
            ortEnvironment = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions()
            // 配置CPU线程数（提升推理速度）
            sessionOptions.setIntraOpNumThreads(4)
            // 可选：启用 GPU 加速（安卓8.0+，需设备支持OpenCL）
            // sessionOptions.addVulkanProvider()
            // 可选：启用 NNAPI 加速（设备NPU）
            // sessionOptions.addNnapiProvider()

            // 加载 assets 中的 ONNX 模型（替换为你的 model.onnx）
            val modelBuffer = loadModelFromAssets("model.onnx")
            ortSession = ortEnvironment!!.createSession(modelBuffer, sessionOptions)

            inferenceResult = "✅ ONNX模型加载成功！"
        } catch (e: Exception) {
            inferenceResult = "❌ 模型加载失败：${e.message}"
            e.printStackTrace()
        }
    }

    // 核心方法2：从 assets 读取 ONNX 模型文件（复用原有方法）
    @Throws(Exception::class)
    private fun loadModelFromAssets(modelName: String): MappedByteBuffer {
        return try {
            val inputStream = assets.open(modelName)
            val bytes = inputStream.readBytes()
            inputStream.close()
            val buffer = ByteBuffer.allocateDirect(bytes.size)
            buffer.order(ByteOrder.nativeOrder())
            buffer.put(bytes)
            buffer.rewind()
            buffer as MappedByteBuffer
        } catch (e: Exception) {
            throw Exception("读取模型文件失败: ${e.message}", e)
        }
    }

    // 加载 assets 中的图片（复用）
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

    // 核心方法3：运行 ONNX 模型推理（替换原 TFLite 推理）
    private fun runModelInference() {
        val session = ortSession
        if (session == null) {
            inferenceResult = "❌ ONNX模型未加载，无法推理"
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
                    runImageToImageONNX(session, srcBitmap)
                }
                outputBitmap = outBmp
                inferenceResult = "✅ 推理成功"
            } catch (e: Exception) {
                inferenceResult = "❌ 推理失败：${e.message}"
                e.printStackTrace()
            }
        }
    }

    // 张量布局枚举（复用）
    private enum class TensorLayout {
        NHWC,
        NCHW
    }

    // 图像张量规格（复用）
    private data class ImageTensorSpec(
        val layout: TensorLayout,
        val batch: Long,
        val height: Long,
        val width: Long,
        val channels: Long
    )

    // 调试：用第一个像素的 RGB 值判断输入输出范围
    private fun debugTensorRange(name: String, floatArray: FloatArray, channels: Int) {
        if (floatArray.isEmpty()) return
        val sb = StringBuilder()
        sb.append("$name 范围: min=${floatArray.minOrNull()}, max=${floatArray.maxOrNull()}")
        if (floatArray.size >= channels) {
            sb.append(", 前$channels 值: ")
            for (i in 0 until channels) {
                sb.append("${floatArray[i]}, ")
            }
        }
        android.util.Log.d("ONNX_DEBUG", sb.toString())
    }

    // 解析 ONNX 张量规格（适配 Long 类型，ONNX 用 Long 表示 shape）
    private fun parseImageTensorSpec(shape: LongArray): ImageTensorSpec {
        if (shape.size != 4) {
            throw IllegalStateException("仅支持 4D 张量，实际为：${shape.contentToString()}")
        }

        val n = shape[0]
        val d1 = shape[1]
        val d2 = shape[2]
        val d3 = shape[3]

        // 判断布局：NHWC [1,H,W,C] 或 NCHW [1,C,H,W]
        val nhwc = (d3 == 1L || d3 == 3L)
        val nchw = (d1 == 1L || d1 == 3L)

        return when {
            nhwc && !nchw -> ImageTensorSpec(TensorLayout.NHWC, n, d1, d2, d3)
            nchw && !nhwc -> ImageTensorSpec(TensorLayout.NCHW, n, d2, d3, d1)
            nhwc && nchw -> ImageTensorSpec(TensorLayout.NHWC, n, d1, d2, d3)
            else -> ImageTensorSpec(TensorLayout.NHWC, n, d1, d2, d3)
        }
    }

    // ONNX 图像到图像推理核心逻辑
    private fun runImageToImageONNX(session: OrtSession, srcBitmap: Bitmap): Bitmap {
        // 获取 ONNX 输入信息
        val inputInfo = session.inputInfo.entries.first()
        val inputName = inputInfo.key
        val inputTensorInfo = inputInfo.value.info as ai.onnxruntime.TensorInfo
        val inputShape = inputTensorInfo.shape // ONNX shape 是 LongArray
        val inputType = inputTensorInfo.type

        // 解析输入规格
        val inputSpec = parseImageTensorSpec(inputShape)
        if (inputSpec.batch != 1L) {
            throw IllegalStateException("仅支持 batch=1，实际为：${inputSpec.batch}")
        }
        if (inputSpec.channels != 3L) {
            throw IllegalStateException("仅支持 3 通道 RGB 输入，实际通道数：${inputSpec.channels}")
        }

        val inH = inputSpec.height.toInt()
        val inW = inputSpec.width.toInt()

        // 调整图片尺寸
        val resized = if (srcBitmap.width != inW || srcBitmap.height != inH) {
            Bitmap.createScaledBitmap(srcBitmap, inW, inH, true)
        } else {
            srcBitmap
        }

        // 转换图片为 ONNX 输入张量
        val inputTensor = when (inputType) {
            OnnxJavaType.FLOAT -> {
                val floatBuffer = bitmapToFloatArray(resized, inputSpec.layout)
                debugTensorRange("输入", floatBuffer, 6)
                val fb = FloatBuffer.wrap(floatBuffer)
                OnnxTensor.createTensor(ortEnvironment!!, fb, inputShape)
            }
            OnnxJavaType.UINT8 -> {
                val uint8Buffer = bitmapToUInt8Array(resized, inputSpec.layout)
                val bb = ByteBuffer.wrap(uint8Buffer)
                OnnxTensor.createTensor(ortEnvironment!!, bb, inputShape)
            }
            else -> throw IllegalStateException("不支持的输入类型：$inputType")
        }

        // 获取输出信息
        val outputInfo = session.outputInfo.entries.first()
        val outputTensorInfo = outputInfo.value.info as ai.onnxruntime.TensorInfo
        val outputShape = outputTensorInfo.shape
        val outputType = outputTensorInfo.type

        // 解析输出规格
        val outputSpec = parseImageTensorSpec(outputShape)
        if (outputSpec.batch != 1L) {
            throw IllegalStateException("仅支持 batch=1 输出，实际为：${outputSpec.batch}")
        }
        if (outputSpec.channels != 1L && outputSpec.channels != 3L) {
            throw IllegalStateException("仅支持 1/3 通道输出，实际通道数：${outputSpec.channels}")
        }

        val outH = outputSpec.height.toInt()
        val outW = outputSpec.width.toInt()
        val outC = outputSpec.channels.toInt()

        // 打印调试信息
        inferenceResult = "input=${inputShape.contentToString()} $inputType (${inputSpec.layout}), output=${outputShape.contentToString()} $outputType (${outputSpec.layout})"

        // 执行 ONNX 推理
        val inputs = mapOf(inputName to inputTensor)
        val outputs = session.run(inputs)
        val outputValue = outputs.get(0)

        // 转换输出为 Bitmap
        val resultBitmap = when (outputType) {
            OnnxJavaType.FLOAT -> {
                val rawValue = (outputValue as OnnxTensor).value
                val floatArray = flattenToFloatArray(rawValue)
                debugTensorRange("输出", floatArray, outC)
                floatArrayToBitmap(floatArray, outW, outH, outC, outputSpec.layout)
            }
            OnnxJavaType.UINT8 -> {
                val rawValue = (outputValue as OnnxTensor).value
                val uint8Array = flattenToByteArray(rawValue)
                uint8ArrayToBitmap(uint8Array, outW, outH, outC, outputSpec.layout)
            }
            else -> throw IllegalStateException("不支持的输出类型：$outputType")
        }

        // 释放资源
        inputTensor.close()
        outputs.close()

        return resultBitmap
    }

    // Bitmap 转 Float 数组（适配 ONNX 输入）
    private fun bitmapToFloatArray(bitmap: Bitmap, layout: TensorLayout): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        return if (layout == TensorLayout.NHWC) {
            // NHWC: [H,W,C] → 展平为 [H*W*C]
            val floatArray = FloatArray(w * h * 3)
            for (i in pixels.indices) {
                val c = pixels[i]
                val baseIdx = i * 3
                floatArray[baseIdx] = ((c shr 16) and 0xFF).toFloat()  // R: 0-255
                floatArray[baseIdx + 1] = ((c shr 8) and 0xFF).toFloat()  // G: 0-255
                floatArray[baseIdx + 2] = (c and 0xFF).toFloat()  // B: 0-255
            }
            floatArray
        } else {
            // NCHW: [C,H,W] → 展平为 [C*H*W]
            val floatArray = FloatArray(3 * w * h)
            val rArray = FloatArray(w * h)
            val gArray = FloatArray(w * h)
            val bArray = FloatArray(w * h)

            for (i in pixels.indices) {
                val c = pixels[i]
                rArray[i] = ((c shr 16) and 0xFF).toFloat()
                gArray[i] = ((c shr 8) and 0xFF).toFloat()
                bArray[i] = (c and 0xFF).toFloat()
            }

            // 拼接 C 维度
            System.arraycopy(rArray, 0, floatArray, 0, w * h)
            System.arraycopy(gArray, 0, floatArray, w * h, w * h)
            System.arraycopy(bArray, 0, floatArray, 2 * w * h, w * h)
            floatArray
        }
    }

    // Bitmap 转 UInt8 数组（适配 ONNX 输入）
    private fun bitmapToUInt8Array(bitmap: Bitmap, layout: TensorLayout): ByteArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        return if (layout == TensorLayout.NHWC) {
            val byteArray = ByteArray(w * h * 3)
            for (i in pixels.indices) {
                val c = pixels[i]
                val baseIdx = i * 3
                byteArray[baseIdx] = ((c shr 16) and 0xFF).toByte()
                byteArray[baseIdx + 1] = ((c shr 8) and 0xFF).toByte()
                byteArray[baseIdx + 2] = (c and 0xFF).toByte()
            }
            byteArray
        } else {
            val byteArray = ByteArray(3 * w * h)
            val rArray = ByteArray(w * h)
            val gArray = ByteArray(w * h)
            val bArray = ByteArray(w * h)

            for (i in pixels.indices) {
                val c = pixels[i]
                rArray[i] = ((c shr 16) and 0xFF).toByte()
                gArray[i] = ((c shr 8) and 0xFF).toByte()
                bArray[i] = (c and 0xFF).toByte()
            }

            System.arraycopy(rArray, 0, byteArray, 0, w * h)
            System.arraycopy(gArray, 0, byteArray, w * h, w * h)
            System.arraycopy(bArray, 0, byteArray, 2 * w * h, w * h)
            byteArray
        }
    }

    private fun flattenToFloatArray(value: Any?): FloatArray {
        if (value == null) return FloatArray(0)
        return when (value) {
            is FloatArray -> value
            is Number -> floatArrayOf(value.toFloat())
            is Array<*> -> {
                val out = ArrayList<Float>()
                for (e in value) {
                    val part = flattenToFloatArray(e)
                    for (v in part) out.add(v)
                }
                out.toFloatArray()
            }
            else -> throw IllegalStateException("无法将输出类型 ${value::class.java.name} 展平成 FloatArray")
        }
    }

    private fun flattenToByteArray(value: Any?): ByteArray {
        if (value == null) return ByteArray(0)
        return when (value) {
            is ByteArray -> value
            is Number -> byteArrayOf(value.toInt().toByte())
            is Array<*> -> {
                val out = ArrayList<Byte>()
                for (e in value) {
                    val part = flattenToByteArray(e)
                    for (v in part) out.add(v)
                }
                val result = ByteArray(out.size)
                for (i in out.indices) result[i] = out[i]
                result
            }
            else -> throw IllegalStateException("无法将输出类型 ${value::class.java.name} 展平成 ByteArray")
        }
    }

    // Float 数组转 Bitmap（适配 ONNX 输出）
    private fun floatArrayToBitmap(
        floatArray: FloatArray,
        width: Int,
        height: Int,
        channels: Int,
        layout: TensorLayout
    ): Bitmap {
        val pixels = IntArray(width * height)
        when (layout) {
            TensorLayout.NHWC -> {
                for (i in pixels.indices) {
                    val baseIdx = i * channels
                    if (channels == 1) {
                        val v = (floatArray[baseIdx] * 255f).toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                    } else {
                        val r = (floatArray[baseIdx] * 255f).toInt().coerceIn(0, 255)
                        val g = (floatArray[baseIdx + 1] * 255f).toInt().coerceIn(0, 255)
                        val b = (floatArray[baseIdx + 2] * 255f).toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
            }
            TensorLayout.NCHW -> {
                val planeSize = width * height
                val rPlane = FloatArray(planeSize)
                val gPlane = if (channels == 3) FloatArray(planeSize) else null
                val bPlane = if (channels == 3) FloatArray(planeSize) else null

                System.arraycopy(floatArray, 0, rPlane, 0, planeSize)
                if (channels == 3) {
                    System.arraycopy(floatArray, planeSize, gPlane!!, 0, planeSize)
                    System.arraycopy(floatArray, 2 * planeSize, bPlane!!, 0, planeSize)
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

    // UInt8 数组转 Bitmap（适配 ONNX 输出）
    private fun uint8ArrayToBitmap(
        byteArray: ByteArray,
        width: Int,
        height: Int,
        channels: Int,
        layout: TensorLayout
    ): Bitmap {
        val pixels = IntArray(width * height)
        when (layout) {
            TensorLayout.NHWC -> {
                for (i in pixels.indices) {
                    val baseIdx = i * channels
                    if (channels == 1) {
                        val v = byteArray[baseIdx].toInt() and 0xFF
                        pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                    } else {
                        val r = byteArray[baseIdx].toInt() and 0xFF
                        val g = byteArray[baseIdx + 1].toInt() and 0xFF
                        val b = byteArray[baseIdx + 2].toInt() and 0xFF
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
            }
            TensorLayout.NCHW -> {
                val planeSize = width * height
                val rPlane = ByteArray(planeSize)
                val gPlane = if (channels == 3) ByteArray(planeSize) else null
                val bPlane = if (channels == 3) ByteArray(planeSize) else null

                System.arraycopy(byteArray, 0, rPlane, 0, planeSize)
                if (channels == 3) {
                    System.arraycopy(byteArray, planeSize, gPlane!!, 0, planeSize)
                    System.arraycopy(byteArray, 2 * planeSize, bPlane!!, 0, planeSize)
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

    // 释放 ONNX 资源（页面销毁时）
    override fun onDestroy() {
        super.onDestroy()
        ortSession?.close()
        ortEnvironment?.close()
    }
}

// Compose UI 组件（仅改名称，逻辑不变）
@Composable
fun ONNXDemoUI(
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
            Text(text = "ONNX 本地推理 Demo")

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

            Button(
                onClick = onInferenceClick,
                modifier = Modifier.padding(top = 20.dp, bottom = 20.dp)
            ) {
                Text(text = "运行模型推理")
            }

            Text(
                text = inferenceResult,
                modifier = Modifier.padding(top = 10.dp)
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ONNXDemoPreview() {
    TFLiteDemoTheme {
        ONNXDemoUI(
            inferenceResult = "预览：模型未加载",
            inputBitmap = null,
            outputBitmap = null,
            onInferenceClick = {}
        )
    }
}