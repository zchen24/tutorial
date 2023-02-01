//
// Qt Mix Audio
// Author: Zihan Chen
// Date: 2023-02-02

#include <QSoundEffect>
#include <QRunnable>
#include <QtCore>
#include <iostream>
#include <filesystem>

using namespace std::chrono;
namespace fs = std::filesystem;

class AudioPlayTask : public QRunnable
{
public:
    AudioPlayTask(int delay)
    : delay_(delay){
        std::string audio_file = fs::current_path().string() + "/example.wav";
        std::cout << "audio file = " << audio_file << "\n";
        se_.setSource(QUrl::fromLocalFile(audio_file.c_str()));
        se_.setLoopCount(2);
    }

    void run() {
        std::cout << "AudioPlayTask start\n";
        std::this_thread::sleep_for(duration<double, std::milli>(delay_ * 1000));
        se_.play();
        // wait for 10 sec
        std::this_thread::sleep_for(duration<double, std::milli>(10000));
        std::cout << "AudioPlayTask end\n";
    }

private:
    QSoundEffect se_;
    int delay_{0};
};


int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    auto task1 = new AudioPlayTask(1);
    QThreadPool::globalInstance()->start(task1);
    auto task2 = new AudioPlayTask(3);
    QThreadPool::globalInstance()->start(task2);
    return QCoreApplication::exec();
}