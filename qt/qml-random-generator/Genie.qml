import QtQuick 2.0
import QtQuick.Controls 2.12

Item {
    property real valMin: 0
    property real valMax: 100

    Column {
        spacing: 15
        width: 300

        Label {
            id: txtRange
            width: parent.width
            anchors.horizontalCenter: parent.horizontalCenter
            horizontalAlignment: Text.AlignHCenter

            font.bold: true
            font.pointSize: 20
            text: qsTr(valMin + ' to ' + valMax)
        }

        RangeSlider {
            id:  slider
            width: parent.width
            anchors.horizontalCenter: parent.horizontalCenter
            from: 0
            to: 100
            first.value: from
            second.value: to
            stepSize: 1

            first.onValueChanged: {
                valMin = Math.round(first.value)
                valMax = Math.round(second.value)
            }

            second.onValueChanged: {
                valMin = Math.round(first.value)
                valMax = Math.round(second.value)
            }
        }

        Button {
            id: btnGenerate
            width: parent.width
            anchors.horizontalCenter: parent.horizontalCenter
            text: "Generate"
            onClicked: {
                console.log('min: ' + valMin + '   max: ' + valMax)
                txtValue.text = Math.floor(Math.random() * (valMax - valMin) + valMin)
            }
        }

        Label {
            id: txtValue
            width: parent.width
            anchors.horizontalCenter: parent.horizontalCenter
            horizontalAlignment: Text.AlignHCenter

            font.bold: true
            font.pointSize: 20
            text: qsTr("00")
        }
    }
}

/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
##^##*/
