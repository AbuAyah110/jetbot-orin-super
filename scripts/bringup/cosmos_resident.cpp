// Resident Cosmos Edge-LLM loop: load engines once, then serve request files.
// Not an HTTP server. Does not open motors. Output is JSON on disk.
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <unistd.h>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{

void write_file(std::filesystem::path const& path, std::string const& body)
{
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    std::ofstream out(tmp);
    if (!out.is_open())
    {
        throw std::runtime_error("failed to open " + tmp.string());
    }
    out << body;
    out.close();
    std::filesystem::rename(tmp, path);
}

bool wait_ready(std::filesystem::path const& path, int timeout_ms)
{
    auto const start = std::chrono::steady_clock::now();
    while (!std::filesystem::exists(path))
    {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
            > timeout_ms)
        {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " --engineDir DIR --multimodalEngineDir DIR --ctrlDir DIR [--maxGenerateLength N]\n";
        return EXIT_FAILURE;
    }

    std::string engineDir;
    std::string multimodalEngineDir;
    std::string ctrlDir;
    int64_t maxGenerateLength = 80;

    for (int i = 1; i < argc; ++i)
    {
        std::string const a = argv[i];
        auto take = [&](std::string& dest) {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("missing value for " + a);
            }
            dest = argv[++i];
        };
        if (a == "--engineDir")
        {
            take(engineDir);
        }
        else if (a == "--multimodalEngineDir")
        {
            take(multimodalEngineDir);
        }
        else if (a == "--ctrlDir")
        {
            take(ctrlDir);
        }
        else if (a == "--maxGenerateLength")
        {
            maxGenerateLength = std::stoll(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown arg: " << a << "\n";
            return EXIT_FAILURE;
        }
    }

    if (engineDir.empty() || multimodalEngineDir.empty() || ctrlDir.empty())
    {
        std::cerr << "--engineDir, --multimodalEngineDir, and --ctrlDir are required\n";
        return EXIT_FAILURE;
    }

    std::filesystem::path const ctrl(ctrlDir);
    std::filesystem::create_directories(ctrl);
    std::filesystem::path const inJson = ctrl / "in.json";
    std::filesystem::path const inReady = ctrl / "in.ready";
    std::filesystem::path const outJson = ctrl / "out.json";
    std::filesystem::path const outReady = ctrl / "out.ready";
    std::filesystem::path const loaded = ctrl / "loaded";
    std::filesystem::path const quit = ctrl / "quit";

    auto pluginHandles = loadEdgellmPluginLib();
    (void)pluginHandles;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::unordered_map<std::string, std::string> emptyLora;
    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    try
    {
        runtime = std::make_unique<rt::LLMInferenceRuntime>(engineDir, multimodalEngineDir, emptyLora, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize runtime: %s", e.what());
        return EXIT_FAILURE;
    }
    if (!runtime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed; continuing without it.");
    }

    write_file(loaded, "ok\n");
    LOG_INFO("Cosmos resident ready at %s pid=%d", ctrlDir.c_str(), static_cast<int>(getpid()));

    while (!std::filesystem::exists(quit))
    {
        if (!std::filesystem::exists(inReady))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        if (!wait_ready(inJson, 2000))
        {
            LOG_ERROR("in.ready without in.json");
            std::filesystem::remove(inReady);
            continue;
        }

        Json outputData;
        outputData["responses"] = Json::array();
        bool ok = false;
        try
        {
            auto parsed = exampleUtils::parseRequestFile(inJson, /*batchSizeOverride=*/1, maxGenerateLength);
            auto& batches = parsed.second;
            for (size_t i = 0; i < batches.size(); ++i)
            {
                rt::LLMGenerationResponse response;
                bool const status = runtime->handleRequest(batches[i], response, stream);
                Json responseJson;
                std::string text;
                if (status && !response.outputTexts.empty())
                {
                    text = response.outputTexts[0];
                    ok = true;
                }
                responseJson["output_text"] = text;
                responseJson["ok"] = status;
                if (!response.finishReasons.empty())
                {
                    responseJson["finish_reason"] = rt::finishReasonName(response.finishReasons[0]);
                }
                outputData["responses"].push_back(responseJson);
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Request failed: %s", e.what());
            outputData["error"] = e.what();
        }

        outputData["ok"] = ok;
        write_file(outJson, outputData.dump(2));
        std::filesystem::remove(inReady);
        write_file(outReady, "ok\n");
        LOG_INFO("Served one request ok=%d", static_cast<int>(ok));
    }

    LOG_INFO("Quit requested; engines stay mapped until process exit.");
    return EXIT_SUCCESS;
}
