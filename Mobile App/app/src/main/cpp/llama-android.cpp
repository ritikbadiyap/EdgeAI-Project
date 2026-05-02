#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"
#include "common.h"

#define LOG_TAG "LlamaAndroidJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static mtmd_context* g_mtmd_ctx = nullptr;
static llama_pos g_n_past = 0;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_llamaapp_LlamaEngine_loadModels(JNIEnv *env, jobject thiz, jstring textModelPath, jstring projectorPath) {
    const char *text_model = env->GetStringUTFChars(textModelPath, nullptr);
    const char *proj_model = env->GetStringUTFChars(projectorPath, nullptr);

    LOGI("Initializing Llama Backend...");
    llama_backend_init();

    // Load Text Model
    LOGI("Loading text model: %s", text_model);
    llama_model_params mparams = llama_model_default_params();
    g_model = llama_model_load_from_file(text_model, mparams);
    if (!g_model) {
        LOGE("Failed to load text model from %s", text_model);
        env->ReleaseStringUTFChars(textModelPath, text_model);
        env->ReleaseStringUTFChars(projectorPath, proj_model);
        return JNI_FALSE;
    } 

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048; // typical context size for mm
    g_ctx = llama_init_from_model(g_model, cparams);
    
    if (!g_ctx) {
        LOGE("Failed to create context for text model");
        llama_free_model(g_model);
        g_model = nullptr;
        env->ReleaseStringUTFChars(textModelPath, text_model);
        env->ReleaseStringUTFChars(projectorPath, proj_model);
        return JNI_FALSE;
    }

    // Load Vision Projector
    LOGI("Loading projector model: %s", proj_model);
    mtmd_context_params clip_cparams = mtmd_context_params_default();
    clip_cparams.use_gpu = false;
    
    g_mtmd_ctx = mtmd_init_from_file(proj_model, g_model, clip_cparams);
    if (!g_mtmd_ctx) {
        LOGE("Failed to load mtmd model from %s", proj_model);
        llama_free(g_ctx);
        g_ctx = nullptr;
        llama_free_model(g_model);
        g_model = nullptr;
        env->ReleaseStringUTFChars(textModelPath, text_model);
        env->ReleaseStringUTFChars(projectorPath, proj_model);
        return JNI_FALSE;
    }

    env->ReleaseStringUTFChars(textModelPath, text_model);
    env->ReleaseStringUTFChars(projectorPath, proj_model);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_example_llamaapp_LlamaEngine_generateDiagnostic(JNIEnv *env, jobject thiz, jstring imagePath, jstring prompt, jboolean isNewSession, jobject callback) {
    const char *img_path = env->GetStringUTFChars(imagePath, nullptr);
    const char *txt_prompt = env->GetStringUTFChars(prompt, nullptr);

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");

    if (!g_model || !g_ctx || !g_mtmd_ctx) {
        LOGE("Contexts not initialized.");
        env->CallVoidMethod(callback, onTokenMethod, env->NewStringUTF("[Error: Models not initialized]"));
        env->ReleaseStringUTFChars(imagePath, img_path);
        env->ReleaseStringUTFChars(prompt, txt_prompt);
        return;
    }

    if (isNewSession) {
        llama_memory_clear(llama_get_memory(g_ctx), true);
        g_n_past = 0;
        LOGI("Processing image: %s", img_path);
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();

    if (isNewSession) {
        mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(g_mtmd_ctx, img_path);
        if (!bmp) {
             LOGE("Failed to load image from %s", img_path);
             env->CallVoidMethod(callback, onTokenMethod, env->NewStringUTF("[Error: Image Load Failed - Check Path]"));
             mtmd_input_chunks_free(chunks);
             env->ReleaseStringUTFChars(imagePath, img_path);
             env->ReleaseStringUTFChars(prompt, txt_prompt);
             return;
        }

        std::string safe_prompt = "USER: ";
        safe_prompt += mtmd_default_marker();
        safe_prompt += "\n";
        safe_prompt += txt_prompt;
        safe_prompt += "\nASSISTANT:";

        mtmd_input_text text;
        text.text = safe_prompt.c_str();
        text.add_special = true;
        text.parse_special = true;

        const mtmd_bitmap * bmps[] = { bmp };

        int32_t res = mtmd_tokenize(g_mtmd_ctx, chunks, &text, bmps, 1);
        if (res != 0) {
            LOGE("Failed to tokenize chunks: %d", res);
            env->CallVoidMethod(callback, onTokenMethod, env->NewStringUTF("[Error: Tokenization Failed]"));
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
            env->ReleaseStringUTFChars(imagePath, img_path);
            env->ReleaseStringUTFChars(prompt, txt_prompt);
            return;
        }
        mtmd_bitmap_free(bmp);
    } else {
        std::string safe_prompt = "USER: ";
        safe_prompt += txt_prompt;
        safe_prompt += "\nASSISTANT:";

        mtmd_input_text text;
        text.text = safe_prompt.c_str();
        text.add_special = false;
        text.parse_special = true;

        int32_t res = mtmd_tokenize(g_mtmd_ctx, chunks, &text, nullptr, 0);
        if (res != 0) {
            LOGE("Failed to tokenize text chunks: %d", res);
            env->CallVoidMethod(callback, onTokenMethod, env->NewStringUTF("[Error: Text Tokenization Failed]"));
            mtmd_input_chunks_free(chunks);
            env->ReleaseStringUTFChars(imagePath, img_path);
            env->ReleaseStringUTFChars(prompt, txt_prompt);
            return;
        }
    }

    llama_pos new_n_past = 0;
    
    if (mtmd_helper_eval_chunks(g_mtmd_ctx, g_ctx, chunks, g_n_past, 0, 512, true, &new_n_past)) {
        LOGE("Failed to eval mtmd chunks");
        env->CallVoidMethod(callback, onTokenMethod, env->NewStringUTF("[Error: Eval chunks failed]"));
        mtmd_input_chunks_free(chunks);
        env->ReleaseStringUTFChars(imagePath, img_path);
        env->ReleaseStringUTFChars(prompt, txt_prompt);
        return;
    }
    
    g_n_past = new_n_past;

    mtmd_input_chunks_free(chunks);

    common_params cparams_sample;
    common_sampler * smpl = common_sampler_init(g_model, cparams_sample.sampling);
    
    llama_batch batch = llama_batch_init(1, 0, 1);
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    
    LOGI("Begin generation...");
    int n_predict = 500;
    for (int i = 0; i < n_predict; i++) {
        llama_token token_id = common_sampler_sample(smpl, g_ctx, -1);
        common_sampler_accept(smpl, token_id, true);

        if (llama_vocab_is_eog(vocab, token_id)) {
            break;
        }

        std::string token_str = common_token_to_piece(g_ctx, token_id);
        
        jstring jToken = env->NewStringUTF(token_str.c_str());
        env->CallVoidMethod(callback, onTokenMethod, jToken);
        env->DeleteLocalRef(jToken);

        common_batch_clear(batch);
        common_batch_add(batch, token_id, g_n_past++, {0}, true);
        if (llama_decode(g_ctx, batch)) {
            LOGE("Failed to decode token");
            env->CallVoidMethod(callback, onTokenMethod, env->NewStringUTF("[Error: decode loop failed]"));
            break;
        }
    }

    llama_batch_free(batch);
    common_sampler_free(smpl);

    LOGI("Generation finished.");

    env->ReleaseStringUTFChars(imagePath, img_path);
    env->ReleaseStringUTFChars(prompt, txt_prompt);
}

JNIEXPORT void JNICALL
Java_com_example_llamaapp_LlamaEngine_freeMemory(JNIEnv *env, jobject thiz) {
    LOGI("Releasing contexts...");
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    if (g_mtmd_ctx) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
    }
    LOGI("Freed.");
}

}
