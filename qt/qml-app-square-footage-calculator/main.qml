import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.12

Window {
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")

    function updateSqft(width, height) {
        lblSqft.text = width * height + "  sqft"
    }

    Column {
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 20

        Label {
            id: lblSqft
            text: qsTr("0 sqft")
            font.bold: true
            font.pointSize: 16
            anchors.horizontalCenter: parent.horizontalCenter
        }

        Grid {
            rows: 2
            spacing: 10
            Label {
                text: qsTr("Width")
                font.bold: true
                font.pointSize: 12
            }
            SpinBox {
                id: spbWidth
                from: 0
                to: 100
                onValueChanged: {
                    console.debug("width: " + value)
                    updateSqft(spbWidth.value, spbHeight.value)
                }
            }
            Label {
                text: qsTr("Height")
                font.bold: true
                font.pointSize: 12
            }
            SpinBox {
                id: spbHeight
                from: 0
                to: 100
                onValueChanged: {
                    console.debug("width: " + value)
                    updateSqft(spbWidth.value, spbHeight.value)
                }
            }
        }
    }
}
