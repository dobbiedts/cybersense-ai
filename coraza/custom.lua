function handle_request()
    local http = require("resty.http")
    local httpc = http.new()
    local cjson = require("cjson")
    
    -- Retrieve request payload safely
    local payload = coraza.request_body() or coraza.request_args() or ""
    
    -- Configure HTTP client timeouts (critical!)
    httpc:set_timeouts(1000, 2000, 2000)  -- Connect, send, read (in ms)
    
    -- ========== XSS Detection ==========
    local xss_ok, xss_err = pcall(function()
        local xss_res, err = httpc:request_uri("http://xss-api:5000/predict", {
            method = "POST",
            body = cjson.encode({ payload = payload }),
            headers = { ["Content-Type"] = "application/json" }
        })

        if not xss_res then
            coraza.log("XSS API connection failed: " .. (err or "unknown error"))
            return
        end

        if xss_res.status == 200 then
            local xss_data = cjson.decode(xss_res.body)
            if xss_data.is_threat then
                coraza.log("XSS threat detected: " .. payload)
                coraza.block_request()
            end
        else
            coraza.log("XSS API returned status: " .. xss_res.status)
        end
    end)
    
    if not xss_ok then
        coraza.log("XSS API error: " .. (xss_err or "unknown error"))
    end
    
    -- ========== SQLi Detection ==========
    local sqli_ok, sqli_err = pcall(function()
        local sql_res, err = httpc:request_uri("http://sql-api:5000/predict_sql", {
            method = "POST",
            body = cjson.encode({ payload = payload }),
            headers = { ["Content-Type"] = "application/json" }
        })

        if not sql_res then
            coraza.log("SQLi API connection failed: " .. (err or "unknown error"))
            return
        end

        if sql_res.status == 200 then
            local sql_data = cjson.decode(sql_res.body)
            if sql_data.is_threat then
                coraza.log("SQLi threat detected: " .. payload)
                coraza.block_request()
            end
        else
            coraza.log("SQLi API returned status: " .. sql_res.status)
        end
    end)
    
    if not sqli_ok then
        coraza.log("SQLi API error: " .. (sqli_err or "unknown error"))
    end
    
    -- Ensure HTTP client is closed
    httpc:close()
end