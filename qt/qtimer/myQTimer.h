
#ifndef MY_QTIMER_H
#define MY_QTIMER_H

#include <QTimer>


class MyQTimer: QObject {
    Q_OBJECT

public:
    MyQTimer();
    virtual ~MyQTimer() = default;

public slots:
    void timer_update();

private:
    QTimer* timer_;
};

#endif // MY_QTIMER_H
