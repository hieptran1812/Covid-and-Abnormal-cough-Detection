
#!/bin/sh

# lệnh build container
# lệnh run container ở detach mode
sleep 2s
docker ps # show container để confirm nó đã chạy

# run test: chạy python script để gọi API endpoints và verify response