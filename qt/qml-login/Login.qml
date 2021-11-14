import QtQuick 2
import QtQuick.Controls 2

Item {
    id: item1
    signal login(string username, string password)
    signal signup()

    Column {
        id: column
        width: 200
        height: 400
        spacing: 10
        anchors.centerIn: parent

        Label {
            id: lblUsername
            text: qsTr("Username")
            font.bold: true
            font.pointSize: 18
            color: "blue"
        }

        TextField {
            id: txtUsername
            placeholderText: qsTr("Text Field")
        }

        Label {
            id: lblPassword
            text: qsTr("Password")
            font.bold: true
            font.pointSize: 18
            color: "red"
        }

        TextField {
            id: txtPassword
            placeholderText: qsTr("Text Field")
            echoMode: TextInput.Password
        }

        Row {
            spacing: 10
            Button {
                id: btnLogin
                text: qsTr("Login")

                onClicked: {
                    console.log("login button clicked")
                    login(txtUsername.text, txtPassword.text)
                }
            }

            Button {
                id: btnSignup
                text: qsTr("Signup")

                onClicked: {
                    console.log("signup button clicked")
                    signup()
                }
            }
        }
    }
}

/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
##^##*/
