FROM nginx:1.25

RUN apt-get update && apt-get install -y \
    libmodsecurity3 \
    modsecurity-crs \
    && rm -rf /var/lib/apt/lists/*

COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY modsecurity /etc/modsecurity/

RUN nginx -t

CMD ["nginx", "-g", "daemon off;"]