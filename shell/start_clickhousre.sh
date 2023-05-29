docker run -d  -p 8123:8123 -p 9000:9000 -p8443:8443 --name opple-clickhouse-server \
-v /dl_app/clickhouse/ch_data:/var/lib/clickhouse/ \
-v /dl_app/clickhouse/ch_logs:/var/log/clickhouse-server/ \
-e CLICKHOUSE_DB=opple_tsdb -e CLICKHOUSE_USER=admin -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 -e CLICKHOUSE_PASSWORD=opuser123456 \
--ulimit nofile=262144:262144 \
--restart always \
clickhouse/clickhouse-server