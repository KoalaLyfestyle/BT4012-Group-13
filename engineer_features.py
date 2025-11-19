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
    # IP Address + Short TLD
    # ------------------------------
    # Legitimate domains usually have standard TLDs (.com, .net, .org, all 3-letter TLDs) and are rarely accessed via IP.
    # URLs with an IP or suspiciously short/missing TLD are highly suspicious.
    final_df['ip_and_short_tld'] = (
        final_df['is_domain_ip'] & (final_df['length_tld'] <= 2)
    ).astype(bool)

    # ------------------------------
    # HTTP + Missing Domain Structure
    # ------------------------------
    # URLs without subdomains or with very short SLD AND using HTTP are unusual.
    # Captures low-information, suspicious URLs with minimal domain structure.
    final_df['http_and_missing_domain_info'] = (
        (final_df['is_http'] == 1) &
        (final_df['has_subdomain'] == 0) &
        (final_df['length_sld'] <= 3)
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
    # - IP + HTTPS is rare but still unusual (not using, zero variance in dataset)
    final_df['ip_x_http'] = (final_df['is_domain_ip'] * final_df['is_http']).astype(bool)
    # final_df['ip_x_https'] = (final_df['is_domain_ip'] * final_df['is_https']).astype(bool)

    # ------------------------------
    # Domain Complexity Score
    # ------------------------------
    # Aggregates multiple weak signals into a single interpretable score:
    # - More subdomains → more suspicious
    # - Unusually short TLD (≤2 chars, likely missing or invalid) → suspicious
    # - IP usage → highly suspicious
    # Higher score → more complex / suspicious domain
    final_df['domain_complexity_score'] = (
        final_df['num_subdomain'] +
        (final_df['length_tld'] <= 2).astype(int) +
        final_df['is_domain_ip'] * 2
    )

    # Normalized domain complexity score (0 to 1 scale)
    max_complexity = 15 + 1 + 2  # max subdomains + short TLD + IP
    # max subdomains generously estimated to avoid data leakage
    final_df['domain_complexity_score'] = final_df['domain_complexity_score'] / max_complexity

    # ------------------------------
    # Suspicion Score
    # ------------------------------
    # Rule-based aggregate of key red flags:
    # - HTTP protocol
    # - Many subdomains
    # - IP usage
    # - Short/missing TLD (≤2 characters)
    # Higher score → more suspicious. Interpretable for stakeholders.
    final_df['suspicion_score'] = (
        final_df['is_http'] * 2 +
        (final_df['num_subdomain'] > 2).astype(int) * 2 +
        final_df['is_domain_ip'].astype(int) * 3 +
        (final_df['length_tld'] <= 2).astype(int) * 2
    )

    # Normalized suspicion score (0 to 1 scale)
    max_score = 9  # Maximum possible suspicion score based on above weights
    final_df['suspicion_score'] = final_df['suspicion_score'] / max_score

    return final_df
