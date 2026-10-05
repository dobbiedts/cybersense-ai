FROM nginx:1.25

# Install ModSecurity
RUN apt-get update && apt-get install -y \
    libmodsecurity3 \
    modsecurity-crs \
    && rm -rf /var/lib/apt/lists/*

# Copy configurations
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY modsecurity/modsecurity.conf /etc/modsecurity/modsecurity.conf



CMD ["nginx", "-g", "daemon off;"]