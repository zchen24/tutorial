// Code is based on the following tutorial
// http://dranger.com/ffmpeg/tutorial03.html
// 2018-06-24
//
// OS: Ubuntu 16.04
// ffmpeg version: 2.8
// SDL version: 2.0.5


#include <iostream>
#include <SDL2/SDL.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}


int main(int argc, char** argv) {
    std::cout << "Hello, World!" << std::endl;
    int ret;
    AVFormatContext* pFormatCtx;

    av_register_all();
    

    int videoStream = -1;
    int audioStream = -1;

    for (int i = 0; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO &&
            videoStream < 0)
        {
            videoStream = i;
        }
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO &&
            audioStream < 0)
        {
            audioStream = i;
        }
    }
    if (videoStream < 0) {
        std::cerr << "Failed to find a video stream\n";
        return -1;
    }
    if (audioStream < 0) {
        std::cerr << "Failed to find an audio stream\n";
        return -1;
    }

    // -----------------------------
    // Get AVCodecContext
    // -----------------------------
    AVCodecContext *aCodecCtxOrig;
    AVCodecContext *aCodecCtx;
    AVCodec *aCodec;

    aCodecCtxOrig = pFormatCtx->streams[audioStream]->codec;
    aCodec = avcodec_find_decoder(aCodecCtxOrig->codec_id);
    if (!aCodec) {
        std::cerr << "Unsupported codec!\n";
        return -1;
    }

    // Copy content
    aCodecCtx = avcodec_alloc_context3(aCodec);
    if (avcodec_copy_context(aCodecCtx, aCodecCtxOrig) != 0) {
        std::cerr << "Failed to copy codec context\n";
        return -1;
    }

    if (avcodec_open2(aCodecCtx, aCodec, nullptr) < 0) {
        std::cerr << "Failed to open codec\n";
    }

    SDL_AudioSpec desired_audio_spec;
    SDL_AudioSpec audio_spec;

    if (SDL_OpenAudio(&desired_audio_spec, &audio_spec) < 0) {
        std::cerr << "Failed: SDL OpenAudio. " << SDL_GetError() << "\n";
    }

    return 0;
}
