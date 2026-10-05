# CyberSense AI

Closed-loop AI platform for real-time threat detection and Cyber Essentials compliance assessment.

## Services
- sql-api (port 5001): SQL injection detection, Random Forest + BiLSTM
- xss-api (port 5000): XSS detection, Random Forest + Attention DNN
- bridge (port 5003): evidence builder and feedback loop
- project-b: compliance engine (in progress)

## Run
docker-compose build sql-api xss-api bridge
docker-compose up sql-api xss-api bridge
