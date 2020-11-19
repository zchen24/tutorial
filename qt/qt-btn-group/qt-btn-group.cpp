//
// C++ Qt Button Group
// 2020-11-19


#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QButtonGroup>
#include <QApplication>

class MyWidget : public QWidget
{
public:
    MyWidget() {
        // IMPORTANT:
        //   use pointer here, as Qt manages them internally (parent pointer)
        //   otherwise, the objects (e.g. pushButton) will be deleted.
        auto* vbox = new QVBoxLayout;
        auto* rb1 = new QPushButton("Option 1");
        auto* rb2 = new QPushButton("Option 2");
        auto* rb3 = new QPushButton("Option 3");
        rb1->setCheckable(true);
        rb2->setCheckable(true);
        rb3->setCheckable(true);

        auto* btn_group = new QButtonGroup(this);
        btn_group->addButton(rb1);
        btn_group->addButton(rb2);
        btn_group->addButton(rb3);

        vbox->addWidget(rb1);
        vbox->addWidget(rb2);
        vbox->addWidget(rb3);
        rb1->setChecked(true);
        setLayout(vbox);
        show();
        startTimer(3000);
    }
    ~MyWidget() override  = default;

public slots:
    void timerEvent(QTimerEvent* event) override {}
};


int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    MyWidget mw;
    return QApplication::exec();
}