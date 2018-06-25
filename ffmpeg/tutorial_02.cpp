// Code is based on the following tutorial 02
// http://dranger.com/ffmpeg/tutorial02.html
// 2018-06-24
//
// Covers the following items:


#include <iostream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_thread.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

int main(int argc, char *argv[])
{
    std::cout << "hello\n";

    av_register_all();

    AVFormatContext *pFormatCtx = nullptr;
    if (avformat_open_input(&pFormatCtx, argv[1], nullptr, nullptr) != 0) {
        std::cerr << "Failed to open file\n";
        return -1;
    }

    if (avformat_find_stream_info(pFormatCtx, nullptr) < 0) {
        std::cerr << "Failed to find stream info\n";
        return -1;
    }

    AVCodecContext *pCodecCtx = nullptr;
    AVCodecContext *pCodecCtxOriginal = nullptr;
    int videoStream = -1;
    for (int i = 0; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStream = i;
            break;
        }
    }
    if (videoStream == -1) {
        std::cerr << "Failed to find a VIDEO stream\n";
        return -1;
    } else {
        std::cout << "Video stream = " << videoStream << "\n";
    }
    pCodecCtxOriginal = pFormatCtx->streams[videoStream]->codec;

    AVCodec *pCodec = nullptr;
    pCodec = avcodec_find_decoder(pCodecCtxOriginal->codec_id);
    if (pCodec == nullptr) {
        std::cerr << "Unsupported codec!\n";
        return -1;
    }

    // Copy Context
    pCodecCtx = avcodec_alloc_context3(pCodec);
    if (avcodec_copy_context(pCodecCtx, pCodecCtxOriginal) != 0) {
        std::cerr << "Failed to copy Codec Context\n";
        return -1;
    }

    // Open Codec
    if (avcodec_open2(pCodecCtx, pCodec, nullptr) < 0) {
        std::cerr << "Failed to open Codec\n";
        return -1;
    }

    AVFrame* pFrame;
    AVFrame* pFrameBGR;
    pFrame = av_frame_alloc();
    pFrameBGR = av_frame_alloc();
    if (pFrame == nullptr || pFrameBGR == nullptr) {
        std::cerr << "Failed to allocate frame\n";
        return -1;
    }

    int numFrameBytes;
    numFrameBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pCodecCtx->width, pCodecCtx->height);
    avpicture_fill((AVPicture*)pFrameBGR,
                   (uint8_t*)av_malloc(numFrameBytes * sizeof(uint8_t)),
                   AV_PIX_FMT_BGR24,
                   pCodecCtx->width,
                   pCodecCtx->height);
    std::cout << "Picture width = " << pCodecCtx->width << " height = " << pCodecCtx->height << "\n";

    // Format Conversion
    struct SwsContext * pSwsCtx = nullptr;
    pSwsCtx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt,
                             pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_BGR24,
                             SWS_BILINEAR,  nullptr, nullptr, nullptr);
    if (pSwsCtx == nullptr) {
        std::cerr << "Failed to get the conversion context\n";
    }



    // ----------------------------------------
    // Tutorial 02: Outputting to the Screen
    // ----------------------------------------
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
        std::cerr << "Could not initialize SDL " << SDL_GetError() << "\n";
        return -1;
    }

    // -----------------------------
    // Loop through the video file
    // -----------------------------

    int packet_index = 0;
    AVPacket packet;
    while (av_read_frame(pFormatCtx, &packet) >= 0) {
        if (packet.stream_index == videoStream) {
            packet_index++;

            // convert to AVFrame
            int got_picture;
            if (avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet) < 0) {
                std::cerr << "Failed to decode video AVPacket\n";
            }

            if (got_picture) {
                std::cout << "Successfully decoded a frame " << packet_index << "\n";

                // convert to RGB format
                sws_scale(pSwsCtx,
                          pFrame->data,
                          pFrame->linesize,
                          0,
                          pFrame->height,
                          pFrameBGR->data,
                          pFrameBGR->linesize);

                cv::Mat mat(pCodecCtx->height, pCodecCtx->width, CV_8UC3, pFrameBGR->data[0]);
                char filename[100];
                sprintf(filename, "frame%02d.jpg", packet_index);
                cv::imwrite(filename, mat);
            }
        }

        // Free the packet that was allocated by av_read_frame
        av_free_packet(&packet);

        if (packet_index == 12) {
            break;
        }
    }

    // Free allocated resources
    av_frame_free(&pFrame);
    av_frame_free(&pFrameBGR);

    // Close the codec
    avcodec_close(pCodecCtx);
    avcodec_close(pCodecCtxOriginal);

    // Close the video file
    avformat_close_input(&pFormatCtx);
    return 0;
}
