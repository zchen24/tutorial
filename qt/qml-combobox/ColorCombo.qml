import QtQuick 2.0
import QtQuick.Controls 2.12
import QtQml.Models 2.12

Item {
    ComboBox {
        id: root
        currentIndex: 1
        width: 200
        model: ["red", "green", "blue", "yellow"]

        delegate: ItemDelegate {
            width: root.width
            highlighted: root.highlightedIndex === index

            contentItem: Row {
                spacing: 5
                width: root.width

                // Draw the color rectangle
                Rectangle {
                    anchors.verticalCenter: parent.verticalCenter
                    width: 10
                    height: 10
                    border.width: 1
                    border.color: "black"
                    color: modelData
                }

                Text {
                    text: qsTr(modelData)
                    color: "black"
                    elide: Text.ElideRight
                    verticalAlignment: Text.AlignVCenter
                }
            }

            background: Rectangle {
                width: root.width
                implicitHeight: 40
                implicitWidth: 100
                border.color: root.currentIndex === index ? "green" : "white"
                color: root.currentIndex === index ? "lightgreen" : "white"
            }
        }

        onCurrentIndexChanged: {
            console.debug("Current Index: " + currentIndex)
        }
    }
}
