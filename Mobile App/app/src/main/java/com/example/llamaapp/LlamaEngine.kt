package com.example.llamaapp

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.FileOutputStream

interface LlamaCallback {
    fun onToken(token: String)
}

object LlamaEngine {
    init {
        System.loadLibrary("llama-android")
    }

    /**
     * Scoped Storage Protection: Copies the selected image to the internal cache directory
     * so that the C++ POSIX native engine can access a physical absolute path.
     */
    fun copyImageToCache(context: Context, uri: Uri): String? {
        return try {
            val file = File(context.cacheDir, "temp_llama_image.jpg")
            context.contentResolver.openInputStream(uri)?.use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            file.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // Native Interface Methods
    external fun loadModels(textModelPath: String, projectorPath: String): Boolean
    external fun generateDiagnostic(imagePath: String, prompt: String, isNewSession: Boolean, callback: LlamaCallback)
    external fun freeMemory()
}
