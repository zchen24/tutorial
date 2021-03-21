// QML for Beginners: Assignment 12
// https://www.udemy.com/course/qml-for-beginners/

import QtQuick 2.12
import QtQuick.Window 2.12

Window {
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")

    Genie {
        width: 300
        anchors.fill: parent
    }
}
