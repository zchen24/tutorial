// JsonCpp Example code from
// http://jsoncpp.sourceforge.net/

#include <iostream>
#include <fstream>
#include <json/json.h>

int main(int argc, char** argv)
{
    Json::Value root;
    Json::Reader reader;
    std::ifstream file("config.json");
    if (!file.is_open()) {
        std::cerr << "file is not open" << std::endl;
        return -1;
    }

    bool rc = reader.parse(file, root);
    if (!rc) {
        std::cerr << "Failed to parse config.json" << std::endl
                  << reader.getFormattedErrorMessages() << std::endl;
        return -1;
    }

    // use get
    std::string encoding = root.get("encoding", "UTF-8").asString();
    std::cout << "encoding: " << encoding << std::endl;

    // use [key] & load array
    const Json::Value plugins = root["plug-ins"];
    if (!plugins.isNull()) {
        std::cout << "--- Loading plugin ---" << std::endl;
        for (int i = 0; i < plugins.size(); i++) {
            std::cout << i << " " << plugins[i].asString() << std::endl;
        }
    }

    // load object
    std::cout << "length = " << root["indent"]["length"] << std::endl;

    // input from stream
//    std::cin >> root["subtree"];

    // output to stream
//    std::cout << root;
}
