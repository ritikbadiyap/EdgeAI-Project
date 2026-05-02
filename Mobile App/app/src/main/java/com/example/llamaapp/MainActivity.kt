package com.example.llamaapp

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings as SettingsIcon
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class ChatMessage(
    val role: String, // "User" or "LLaMA"
    val content: String,
    val imageUri: Uri? = null,
    val metrics: String? = null
)

class MainViewModel : ViewModel(), LlamaCallback {
    var messages = mutableStateListOf<ChatMessage>()
    var isGenerating by mutableStateOf(false)
        private set
    var isModelLoaded by mutableStateOf(false)
        private set
    var selectedModel by mutableStateOf("4-bit (Fast)")
    val availableModels = listOf("4-bit (Fast)", "8-bit (Balanced)", "F16 (Accurate)", "1.4B Base")

    var isContinuousChat by mutableStateOf(true)

    // Live Metrics
    var currentTokSec by mutableStateOf(0f)
    var currentEvalMs by mutableStateOf(0L)
    var peakRamMb by mutableStateOf(0L)
    var ttftMs by mutableStateOf(0L)

    private var tokenCount = 0
    private var startTimeMs = 0L
    private var firstTokenTimeMs = 0L

    private fun getModelsDir(): File {
        return File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "BananaVLM_Models")
    }

    fun loadModel(context: Context) {
        if (isGenerating) return
        isGenerating = true
        viewModelScope.launch(Dispatchers.IO) {
            val baseDir = getModelsDir()
            if (!baseDir.exists()) {
                withContext(Dispatchers.Main) {
                    messages.add(ChatMessage("System", "Error: Directory not found: ${baseDir.absolutePath}"))
                    isGenerating = false
                }
                return@launch
            }

            val projectorPath = File(baseDir, "mmproj-model-f16.gguf").absolutePath
            val textModelPath = when(selectedModel) {
                "4-bit (Fast)" -> File(baseDir, "Banana_mobilevlm_q4_k_m.gguf").absolutePath
                "8-bit (Balanced)" -> File(baseDir, "Banana_mobilevlm_q8_0.gguf").absolutePath
                "F16 (Accurate)" -> File(baseDir, "Banana_mobilevlm_f16.gguf").absolutePath
                else -> File(baseDir, "MobileLLaMA-1.4B-Base-F32.gguf").absolutePath
            }

            if (!File(textModelPath).exists() || !File(projectorPath).exists()) {
                withContext(Dispatchers.Main) {
                    messages.add(ChatMessage("System", "Error: Model files not found in ${baseDir.absolutePath}"))
                    isGenerating = false
                }
                return@launch
            }

            try {
                LlamaEngine.freeMemory()
                val success = LlamaEngine.loadModels(textModelPath, projectorPath)
                withContext(Dispatchers.Main) {
                    if (success) {
                        isModelLoaded = true
                        messages.add(ChatMessage("System", "Models loaded successfully! Ready to chat."))
                    } else {
                        isModelLoaded = false
                        messages.add(ChatMessage("System", "Error: Native model initialization failed."))
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { messages.add(ChatMessage("System", "Error loading models: ${e.message}")) }
            } finally {
                withContext(Dispatchers.Main) { isGenerating = false }
            }
        }
    }

    fun diagnose(context: Context, imagePath: String, prompt: String, imageUri: Uri?) {
        if (isGenerating || !isModelLoaded) return
        isGenerating = true
        
        val isNewSession = !isContinuousChat || messages.isEmpty() || messages.lastOrNull()?.role == "System"

        messages.add(ChatMessage("User", prompt, if (isNewSession) imageUri else null))
        messages.add(ChatMessage("LLaMA", ""))

        val messageIndex = messages.size - 1
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        
        tokenCount = 0
        peakRamMb = 0L
        startTimeMs = System.currentTimeMillis()
        firstTokenTimeMs = 0L
        ttftMs = 0L

        viewModelScope.launch(Dispatchers.IO) {
            val ramPoller = launch(Dispatchers.Default) {
                while (true) {
                    try {
                        val pids = intArrayOf(android.os.Process.myPid())
                        val memInfo = activityManager.getProcessMemoryInfo(pids)
                        val ram = memInfo[0].totalPss / 1024L
                        if (ram > peakRamMb) {
                            withContext(Dispatchers.Main) { peakRamMb = ram }
                        }
                    } catch (e: Exception) {}
                    delay(500)
                }
            }

            try {
                LlamaEngine.generateDiagnostic(imagePath, prompt, isNewSession, this@MainViewModel)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    messages[messageIndex] = messages[messageIndex].copy(content = messages[messageIndex].content + "\nError: ${e.message}")
                }
            } finally {
                val endTimeMs = System.currentTimeMillis()
                ramPoller.cancel()
                withContext(Dispatchers.Main) {
                    val finalEvalMs = endTimeMs - startTimeMs
                    val decodeTimeMs = endTimeMs - firstTokenTimeMs
                    val finalTokSec = if (decodeTimeMs > 0 && tokenCount > 1) {
                        ((tokenCount - 1) * 1000f) / decodeTimeMs
                    } else {
                        0f
                    }
                    val finalMetrics = "⏱ TTFT: ${ttftMs}ms | ⚡ ${String.format("%.1f", finalTokSec)} tok/s | ⏱ Total: ${finalEvalMs}ms | 🧠 RAM Peak: ${peakRamMb}MB"
                    messages[messageIndex] = messages[messageIndex].copy(metrics = finalMetrics)
                    isGenerating = false
                    currentTokSec = 0f
                    currentEvalMs = 0L
                    ttftMs = 0L
                }
            }
        }
    }

    override fun onToken(token: String) {
        viewModelScope.launch(Dispatchers.Main) {
            val now = System.currentTimeMillis()
            if (tokenCount == 0) {
                firstTokenTimeMs = now
                ttftMs = firstTokenTimeMs - startTimeMs
            }
            tokenCount++
            currentEvalMs = now - startTimeMs

            val decodeTimeMs = now - firstTokenTimeMs
            currentTokSec = if (decodeTimeMs > 0 && tokenCount > 1) {
                ((tokenCount - 1) * 1000f) / decodeTimeMs
            } else {
                0f
            }

            val lastMsg = messages.last()
            messages[messages.size - 1] = lastMsg.copy(content = lastMsg.content + token)
        }
    }

    override fun onCleared() {
        super.onCleared()
        Thread { LlamaEngine.freeMemory() }.start()
    }
}

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                val viewModel: MainViewModel = viewModel()
                val context = LocalContext.current
                var showSettings by remember { mutableStateOf(false) }

                Scaffold(
                    topBar = {
                        TopAppBar(
                            title = { Text("Llama Multimodal Chat") },
                            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                            actions = {
                                IconButton(onClick = { showSettings = true }) {
                                    Icon(Icons.Default.SettingsIcon, contentDescription = "Settings")
                                }
                            }
                        )
                    }
                ) { padding ->
                    ChatScreen(viewModel, context, padding)
                    
                    if (showSettings) {
                        SettingsModal(viewModel, context, onDismiss = { showSettings = false })
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsModal(viewModel: MainViewModel, context: Context, onDismiss: () -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Configuration Details") },
        text = {
            Column(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                if (!Environment.isExternalStorageManager()) {
                    Button(
                        onClick = {
                            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                            intent.data = Uri.parse("package:${context.packageName}")
                            context.startActivity(intent)
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                    ) {
                        Text("Grant Models Folder Access")
                    }
                }
                
                ExposedDropdownMenuBox(
                    expanded = expanded,
                    onExpandedChange = { expanded = !expanded }
                ) {
                    OutlinedTextField(
                        value = viewModel.selectedModel,
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Quality Priority") },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                        modifier = Modifier.menuAnchor().fillMaxWidth()
                    )
                    ExposedDropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false }
                    ) {
                        viewModel.availableModels.forEach { modelName ->
                            DropdownMenuItem(
                                text = { Text(modelName) },
                                onClick = {
                                    viewModel.selectedModel = modelName
                                    expanded = false
                                }
                            )
                        }
                    }
                }
                
                Button(
                    onClick = { 
                        viewModel.loadModel(context)
                        onDismiss()
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !viewModel.isGenerating
                ) {
                    Text("Load Selected Model")
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        }
    )
}

@Composable
fun ChatScreen(viewModel: MainViewModel, context: Context, padding: PaddingValues) {
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var cachedImagePath by remember { mutableStateOf<String?>(null) }
    var prompt by remember { mutableStateOf("") }
    
    val listState = rememberLazyListState()

    val photoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri ->
            if (uri != null) {
                imageUri = uri
                cachedImagePath = LlamaEngine.copyImageToCache(context, uri)
                // Force a new session when picking a new image
                viewModel.isContinuousChat = false
            }
        }
    )

    LaunchedEffect(viewModel.messages.size, viewModel.messages.lastOrNull()?.content) {
        if (viewModel.messages.isNotEmpty()) {
            listState.animateScrollToItem(viewModel.messages.size - 1)
        }
    }

    Column(
        modifier = Modifier
            .padding(padding)
            .fillMaxSize()
            .background(Color(0xFFF5F5F5))
    ) {
        // Chat History List
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(16.dp),
            state = listState,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(viewModel.messages) { msg ->
                ChatBubble(msg)
            }
        }

        // Live Dynamic Metrics
        if (viewModel.isGenerating && viewModel.currentEvalMs > 0) {
            val ttftStr = if (viewModel.ttftMs > 0) "⏱ TTFT: ${viewModel.ttftMs}ms | " else ""
            Text(
                text = "⚡ Generating... ${ttftStr}${String.format("%.1f", viewModel.currentTokSec)} tok/s | ⏱ Total: ${viewModel.currentEvalMs}ms | 🧠 ${viewModel.peakRamMb} MB",
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp),
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.bodySmall
            )
        }

        // Input Area
        Surface(
            color = MaterialTheme.colorScheme.surface,
            shadowElevation = 8.dp,
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                // Controls Option Bar
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = { photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                    ) {
                        Text("Pick Image")
                    }
                    
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Keep Session", style = MaterialTheme.typography.bodySmall)
                        Checkbox(
                            checked = viewModel.isContinuousChat,
                            onCheckedChange = { viewModel.isContinuousChat = it }
                        )
                    }
                }
                
                if (imageUri != null && (!viewModel.isContinuousChat || viewModel.messages.isEmpty())) {
                    Text("Image attached.", color = MaterialTheme.colorScheme.secondary, style = MaterialTheme.typography.bodySmall)
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = prompt,
                        onValueChange = { prompt = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text("Ask something...") },
                        maxLines = 3
                    )
                    
                    Button(
                        onClick = {
                            if (cachedImagePath != null) {
                                viewModel.diagnose(context, cachedImagePath!!, prompt, imageUri)
                                prompt = ""
                            }
                        },
                        enabled = !viewModel.isGenerating && viewModel.isModelLoaded && cachedImagePath != null && prompt.isNotBlank()
                    ) {
                        Text("Send")
                    }
                }
            }
        }
    }
}

@Composable
fun ChatBubble(msg: ChatMessage) {
    val isUser = msg.role == "User"
    val align = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
    val bg = if (isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.secondaryContainer
    val fg = if (isUser) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSecondaryContainer
    
    Box(modifier = Modifier.fillMaxWidth(), contentAlignment = align) {
        Column(
            modifier = Modifier
                .background(bg, RoundedCornerShape(12.dp))
                .padding(12.dp)
                .fillMaxWidth(0.85f)
        ) {
            Text(text = msg.role, fontWeight = FontWeight.Bold, color = fg.copy(alpha = 0.7f), fontSize = MaterialTheme.typography.labelMedium.fontSize)
            
            if (msg.imageUri != null) {
                Spacer(modifier = Modifier.height(4.dp))
                AsyncImage(
                    model = msg.imageUri,
                    contentDescription = "Chat Image Attachment",
                    modifier = Modifier.fillMaxWidth().height(150.dp)
                )
            }
            
            Spacer(modifier = Modifier.height(4.dp))
            SelectionContainer {
                Text(text = msg.content, style = MaterialTheme.typography.bodyLarge, color = fg)
            }
            
            if (msg.metrics != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = msg.metrics,
                    style = MaterialTheme.typography.bodySmall,
                    color = fg.copy(alpha = 0.8f)
                )
            }
        }
    }
}
