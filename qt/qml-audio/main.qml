import QtQuick 2.12
import QtQuick.Controls 2.5
import QtMultimedia 5.12

ApplicationWindow {
    id: window
    width: 640
    height: 480
    visible: true
    title: qsTr("Stack")

    Audio {
        id: player
        volume: 0.5
        autoLoad: true
        source: "qrc:/CantinaBand3.wav"
    }

    Row {
        width: parent.width
        height: parent.height
        spacing: 20

        Button {
            width: 100
            height: 100
            text: "Prev"
            onClicked: {
                console.log("prev 1 sec")
                if (player.seekable) {
                    player.seek(player.position - 1000)
                }
            }
        }
        Button {
            width: 100
            height: 100
            text: (player.playbackState == Audio.PlayingState) ? "Pause" : "Play"
            onClicked: {
                if (player.playbackState == Audio.PlayingState) {
                    console.log("pause")
                    player.pause()
                }
                else {
                    console.log("play")
                    player.play()
                }
            }
        }
        Button {
            width: 100
            height: 100
            text: "Next"
            onClicked: {
                console.log("next 1 sec")
                if (player.seekable) {
                    player.seek(player.position + 1000)
                }
            }
        }
    }
}
