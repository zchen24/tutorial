
#include <iostream>
#include <QApplication>
#include "myQTimer.h"


MyQTimer::MyQTimer()
{
    std::cout << "Hello MyQTTimer\n";

    timer_ = new QTimer(this);
    connect(timer_, SIGNAL(timeout()), this, SLOT(timer_update()));
    timer_->start(10);
}

void MyQTimer::timer_update()
{
    std::cout << "Timer triggered\n";
}


int main(int argc, char** argv)
{
    QApplication app(argc, argv);    
    MyQTimer mt;
    return app.exec();
}


