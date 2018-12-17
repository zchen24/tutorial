import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls 1.4


Window {
    visible: true
    width: 320
    height: 480
    title: "qml state example"

    MouseArea {
        anchors.fill: parent
        onClicked: {
            Qt.quit();
        }
    }

    Column {
        spacing: 10
        anchors.centerIn: parent

        Button {
            id: btnStop
            text: "Stop"
            onClicked: {
                console.log("button stop clicked")
                traffic.state = "stop"
            }
        }

        Button {
            id: btnGo
            text: "Go"
            onClicked: {
                console.log("button go clicked")
                traffic.state = "go"
            }
        }

        TrafficLight {
            id: traffic
        }

    }
}

