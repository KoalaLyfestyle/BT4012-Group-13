def engineer_features(df):

    final_df = df.copy()

    # ------------------------------
    # HTTP + Many Subdomains
    # ------------------------------
    # Attackers often use many nested subdomains to mimic legitimate domains
    # combined with unsecured HTTP. Real websites rarely have both.
    final_df['is_http_and_many_subdomains'] = (
        (final_df['is_http'] == 1) & (final_df['num_subdomain'] > 2)
    ).astype(bool)

    # ------------------------------
    # IP Address + Missing TLD
    # ------------------------------
    # Legitimate domains usually have TLDs and are rarely accessed via IP.
    # URLs with an IP but no TLD are highly suspicious.
    final_df['ip_and_no_tld'] = (
        final_df['is_domain_ip'] & (final_df['has_tld'] == 0)
    ).astype(bool)

    # ------------------------------
    # HTTP + Missing Domain Structure
    # ------------------------------
    # URLs without subdomains or TLDs AND using HTTP are unusual.
    # Captures low-information, suspicious URLs.
    final_df['http_and_missing_domain_info'] = (
        (final_df['is_http'] == 1) &
        (final_df['has_subdomain'] == 0) &
        (final_df['has_tld'] == 0)
    ).astype(bool)

    # ------------------------------
    # Subdomain Depth × HTTP
    # ------------------------------
    # Multiplies subdomain depth by HTTP usage to capture the intensity of suspicion.
    # Deep subdomains are more suspicious if served over HTTP.
    final_df['subdomain_depth_x_http'] = (
        final_df['num_subdomain'] * final_df['is_http']
    )

    # ------------------------------
    # IP × Protocol interactions
    # ------------------------------
    # Using an IP instead of a hostname is suspicious.
    # Combining with protocol amplifies signal:
    # - IP + HTTP is especially suspicious
    # - IP + HTTPS is rare but still unusual
    final_df['ip_x_http'] = final_df['is_domain_ip'] * final_df['is_http']
    final_df['ip_x_https'] = final_df['is_domain_ip'] * final_df['is_https']

    # ------------------------------
    # Domain Complexity Score
    # ------------------------------
    # Aggregates multiple weak signals into a single interpretable score:
    # - More subdomains → more suspicious
    # - Missing TLD → suspicious
    # - IP usage → highly suspicious
    # Higher score → more complex / suspicious domain
    final_df['domain_complexity_score'] = (
        final_df['num_subdomain'] +
        (1 - final_df['has_tld']) +
        final_df['is_domain_ip'] * 2
    )

    # ------------------------------
    # Suspicion Score
    # ------------------------------
    # Rule-based aggregate of key red flags:
    # - HTTP protocol
    # - Many subdomains
    # - IP usage
    # - Missing TLD
    # Higher score → more suspicious. Interpretable for stakeholders.
    final_df['suspicion_score'] = (
        final_df['is_http'] * 2 +
        (final_df['num_subdomain'] > 2).astype(int) * 2 +
        final_df['is_domain_ip'].astype(int) * 3 +
        (final_df['has_tld'] == 0).astype(int) * 2
    )

    return final_df
