package com.twinforge.capture

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Chunked upload to the orchestrator's `POST /capture`.
 *
 * The server keys chunks by index, so re-sending an index overwrites it — a retry after
 * a dropped connection is safe, and a scan interrupted mid-room resumes rather than
 * restarting.
 */
class Uploader(private val baseUrl: String = BuildConfig.ORCHESTRATOR_URL) {

    private val client = OkHttpClient.Builder()
        .callTimeout(30, TimeUnit.SECONDS)
        .build()

    /** The server mints the scan id on the first chunk; every later chunk reuses it. */
    var scanId: String? = null
        private set

    fun uploadFrame(index: Int, npz: ByteArray) {
        val url = StringBuilder("$baseUrl/capture?index=$index")
        scanId?.let { url.append("&scan_id=$it") }

        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file", "frame_$index.npz",
                npz.toRequestBody("application/octet-stream".toMediaType()),
            )
            .build()

        client.newCall(Request.Builder().url(url.toString()).post(body).build()).execute()
            .use { response ->
                val text = response.body?.string().orEmpty()
                if (!response.isSuccessful) {
                    throw java.io.IOException("upload of chunk $index failed: ${response.code} $text")
                }
                scanId = JSONObject(text).getString("scan_id")
            }
    }

    fun complete(): Int {
        val id = scanId ?: throw IllegalStateException("nothing uploaded yet")
        val empty = ByteArray(0).toRequestBody(null)
        client.newCall(
            Request.Builder().url("$baseUrl/capture/$id/complete").post(empty).build()
        ).execute().use { response ->
            val text = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw java.io.IOException("complete failed: ${response.code} $text")
            }
            return JSONObject(text).getInt("frame_count")
        }
    }
}
