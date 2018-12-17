import QtQuick 2.0

Rectangle {
    id: root
    width: 100
    height: 250
    color: "yellow"
    border.color: "black"
    border.width: 1

    gradient: Gradient {
        GradientStop { position: 0.0; color: "darkgrey" }
        GradientStop { position: 1.0; color: "lightgrey" }
    }

    Column {
        spacing: 10
        anchors.centerIn: parent

        Light {
            id: redLight
            width: 70
        }

        Light {
            id: greenLight
            width: 70
        }
    }

    states: [
        State {
            name: "stop"
            PropertyChanges {target: redLight; color: "red"}
            PropertyChanges {target: greenLight; color: "darkgreen"}
        },
        State {
            name: "go"
            PropertyChanges {target: redLight; color: "darkred"}
            PropertyChanges {target: greenLight; color: "lime"}
        }
    ]

    transitions: [
        Transition {
            from: "*"; to: "*"
            ColorAnimation { target: redLight; properties: "color"; duration: 1000 }
            ColorAnimation { target: greenLight; properties: "color"; duration: 1000 }
        }
    ]
}

