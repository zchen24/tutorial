// 配置CLion管理Qt项目国际化支持
// From https://www.cnblogs.com/apocelipes/p/14355460.html

#include <QApplication>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QPushButton>
#include <QWidget>
#include <QTranslator>

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QTranslator trans;
//  if (trans.load("./lang/en_US.qm")) {
//    QCoreApplication::installTranslator(&trans);
//  }

  if (trans.load("./lang/zh_CN.qm")) {
    QCoreApplication::installTranslator(&trans);
  }

  QWidget window;
  auto btn1 = new QPushButton{QObject::tr("click left button")};
  QObject::connect(btn1, &QPushButton::clicked, [w = &window]() {
    QMessageBox::information(w,
                             QObject::tr("clicked left button"),
                             QObject::tr("you clicked left button"));
  });
  auto btn2 = new QPushButton{QObject::tr("click right button")};
  QObject::connect(btn2, &QPushButton::clicked, [w = &window]() {
    QMessageBox::information(w,
                             QObject::tr("clicked right button"),
                             QObject::tr("you clicked right button"));
  });
  auto mainLayout = new QHBoxLayout;
  mainLayout->addWidget(btn1);
  mainLayout->addWidget(btn2);
  window.setLayout(mainLayout);
  window.show();
  return QApplication::exec();
}