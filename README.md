# ArtXFashion


### Installieren und starten der Anwendung
Zuerst müssen Sie Docker installieren. Downloaden können Sie Docker [hier](https://docs.docker.com/get-docker/).

Nachdem Sie Docker installiert haben, müssen Sie unseren Code herunterladen. Dafür gibt es zwei Möglichkeiten:
1. Möglichkeit: Klicken Sie auf den Grünen Knopf "Code", dann auf "Download ZIP". Entpacken Sie die Datei in einem Ordner Ihrer Wahl.
2. Möglichkeit: Falls Sie Git auf Ihrem Computer installiert haben, können Sie in einem Ordner Ihrere Wahl eine Git-Konsole öffnen und das Kommando
  ```git clone git@github.com:u-adrian/CodingDaVinci-ArtXFashion.git``` ausführen. Git wird dann das Repository herunterladen.
  
 Da die Gewichte-Dateien unserer Neuronalen Netze sehr groß sind, haben wir diese nicht auf GitHub hochgeladen.
 Die Gewichte können Sie [hier](https://artxfashion-hackathon.s3.eu-central-1.amazonaws.com/weights_E1000.pt) herunterladen. Speichern Sie die Datei in "CodingDaVinci-ArtXFashion\django-webserver\imageupload\segmentation". Der Name
 der Datei muss "weights_E1000.pt" sein.
 
 
 Falls Sie keine CUDA fähige Grafikkarte haben müssen Sie folgende Zeilen aus der Datei 
 "CodingDaVinci-ArtXFashion/django-webserver/docker-compose.yml" löschen:
 ```
      deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
 ```
 Dies ermöglicht es Ihnen die Anwendung zu installieren.
 
 **DISCLAIMER**: Eine Ausführung ohne CUDA fähige Grafikkarte wird jedoch nicht empfohlen, die Berechnungszeit kann abhängig von der vorhandenen CPU 30 Minuten deutlich übersteigen. Mit performanter Grafikkarte liegt die Berechnungszeit bei ca. einer Minute.
 
  
 Im nächsten Schritt wird die Anwendung installiert:
 
 Dafür öffnen Sie eine Konsole in dem Ordner "CodingDaVinci-ArtXFashion/django-webserver".
 Führen sie das Kommando 
 
 ```docker-compose build```
 aus.
 Sobald das Kommando erfolgreich beendet wurde, müssen Sie 
 
 ```./run_commands.sh```
 ausführen.
 
 Nun können Sie die Anwendung starten.
 Führen Sie dafür in der bereits geöffneten Konsole 
 
 ```docker-compose up```
 aus.
 
 Öffnen Sie einen Browser und öffnen Sie den Link [127.0.0.1:8000/upload](127.0.0.1:8000/upload).
 
 Die Berechnungsziet bei Erstausführung der Anwendung ist gewöhnlich etwas höher, da bestimmte Funktionalitäten aus dem Internet nachgeladen werden müssen.
 
 Viel Spaß beim designen und herumprobieren!
