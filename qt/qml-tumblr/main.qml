import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.12

Window {
    id: window
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")

    property var lockcode: "529"
    property string status: "Locked"

    Popup {
        id: statusPopup
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        width: 200
        height: 300
        modal: true
        focus: true

        Label {
            id: statusLabel
            anchors.centerIn: parent
            text: status
            font.bold: true
        }
    }

    Column {
        id: column
        x: 220
        y: 29
        width: 200
        height: 352
        anchors.verticalCenter: parent.verticalCenter
        spacing: 25
        anchors.horizontalCenter: parent.horizontalCenter

        Label {
            id: label
            text: qsTr("000")
            horizontalAlignment: Text.AlignHCenter
            anchors.horizontalCenter: parent.horizontalCenter
            font.bold: true
            font.pointSize: 15
        }

        Row {
            id: row
            y: 50
            width: 200
            height: 300

            Tumbler {
                id: tumbler
                model: 10
                onCurrentIndexChanged: label.text = tumbler.currentIndex + " " + tumbler1.currentIndex + " " + tumbler2.currentIndex
            }

            Tumbler {
                id: tumbler1
                model: 10
                onCurrentIndexChanged: label.text = tumbler.currentIndex + " " + tumbler1.currentIndex + " " + tumbler2.currentIndex
            }

            Tumbler {
                id: tumbler2
                model: 10
                onCurrentIndexChanged: label.text = tumbler.currentIndex + " " + tumbler1.currentIndex + " " + tumbler2.currentIndex
            }
        }

        Button {
            id: button
            y: 300
            text: qsTr("Check")
            anchors.horizontalCenter: parent.horizontalCenter
            onClicked: {
                var tmpCode = tumbler.currentIndex + "" + tumbler1.currentIndex + "" + tumbler2.currentIndex
                if (tmpCode === lockcode) status = "Unlocked"
                else status = "Locked"
                statusPopup.open()
            }
        }
    }

}
