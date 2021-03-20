import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.15

Window {
    id: window
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")

    Column {
        id: column
        x: 220
        y: 63
        width: 200
        height: 400
        anchors.verticalCenter: parent.verticalCenter
        spacing: 25
        anchors.horizontalCenter: parent.horizontalCenter

        Label {
            id: lblStatus
            text: qsTr("Status")
            font.bold: true
            font.pointSize: 25
            anchors.topMargin: 0
            anchors.horizontalCenter: parent.horizontalCenter
        }

        NumPad {
            id: numPad
            width: 100
            height: 100
            anchors.horizontalCenter: parent.horizontalCenter
            onClicked: lblStatus.text = value
        }
    }


}
