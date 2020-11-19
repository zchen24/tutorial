//
// C++ Qt5 style, connect Signal/Slot
// 2020-11-19
//


#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QButtonGroup>
#include <QApplication>
#include <iostream>

class MyWidget : public QWidget
{
public:
    MyWidget() {
        auto* vbox = new QVBoxLayout;
        auto* pb = new QPushButton("Option 1");
        vbox->addWidget(pb);
        setLayout(vbox);

        // QT5 style
        connect(pb, &QPushButton::clicked, this, &MyWidget::on_pushButton_clicked);
        show();
    }
    ~MyWidget() override  = default;

public slots:
    static void on_pushButton_clicked(bool checked) {
        std::cout << "PushButton Clicked: " << checked << "\n";
    };
    void timerEvent(QTimerEvent* event) override {}
};


int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    MyWidget mw;
    return QApplication::exec();
}