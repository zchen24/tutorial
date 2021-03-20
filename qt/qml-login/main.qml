import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.12

Window {
    width: 640
    height: 480
    visible: true
//    visibility: "FullScreen"
    title: qsTr("Demo: Login Form")

    Popup {
        id: popup
        width: 200
        height: 150
        modal: true
        focus: true
        anchors.centerIn: parent
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        Label {
            id: lblStatus
            anchors.centerIn: parent
            text: ""
        }
    }

    Login {
        id: login
        anchors.centerIn: parent

        onLogin: {
            console.warn("login usr: " + username + " pwd: " + password)
            lblStatus.text = username
            popup.open()
        }
    }
}
