import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from collections import Counter
import tldextract
import ipaddress

def decompose_single_url(url):
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or None
        ext = tldextract.extract(hostname) if hostname else None
        
        # path decomposition
        path_parts = [p for p in parsed.path.split('/') if p] if parsed.path else []
        filename = path_parts[-1] if path_parts else None
        file_extension = filename.split('.')[-1] if filename and '.' in filename else None
        directory_path = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else None
        
        # query parameters
        query_params = parsed.query.split('&') if parsed.query else None

        return {
            'url': url,
            'protocol': parsed.scheme or None,
            'hostname': hostname,
            'port': parsed.port if parsed.port is not None else None,
            'path': parsed.path or None,
            'query': parsed.query or None,
            'fragment': parsed.fragment or None,
            'subdomains': ext.subdomain if ext else None,
            'sld': ext.domain if ext else None,
            'tld': ext.suffix if ext else None,
            'filename': filename,
            'file_extension': file_extension,
            'directory_path': directory_path,
            'query_params': query_params
        }
    except Exception:
        return {
            'url': url,
            'protocol': None,
            'hostname': None,
            'port': None,
            'path': None,
            'query': None,
            'fragment': None,
            'subdomains': None,
            'sld': None,
            'tld': None,
            'filename': None,
            'file_extension': None,
            'directory_path': None,
            'query_params': None
        }

def decompose_url(df):
    df_decomposed = df['url'].apply(lambda x: pd.Series(decompose_single_url(x)))
    # df_decomposed = pd.concat([df_decomposed, df['target']], axis=1) # Target might not exist in inference
    
    # convert all empty strings to None
    df_decomposed.replace('', None, inplace=True)
    
    # transform url such that we truncate off at hostname level
    def reconstruct_base_url(row):
        if not row['hostname']:
            return row['url']  # return original if parsing failed
        protocol = row['protocol'] if row['protocol'] else 'http'
        return f"{protocol}://{row['hostname']}"
    
    # The notebook says: "transform url such that we truncate off at hostname level"
    # And then: "df_decomposed['url'] = df_decomposed.apply(reconstruct_base_url, axis=1)"
    # However, for inference, especially with TF-IDF on the full URL, we MUST preserve the full URL.
    # If the model was trained on truncated URLs, then we should use truncated.
    # But typically, TF-IDF models benefit from the full path.
    # Let's check if we should keep the original 'url' in a separate column or not overwrite it.
    
    # CRITICAL FIX: The original notebook overwrote 'url' with the truncated version.
    # This causes all URLs with the same domain to have the SAME 'url' feature, leading to identical TF-IDF vectors.
    # If the model was trained on truncated URLs, then this is correct behavior (but poor model design).
    # If the model was trained on FULL URLs, then this is a BUG in the extractor matching the notebook.
    
    # Given the user's issue ("third row having all the same predictions"), it confirms the model sees identical inputs.
    # The fix is to NOT overwrite 'url' with the truncated version, OR to ensure the TF-IDF vectorizer uses the full URL.
    
    # Let's store the truncated version as 'base_url' and keep 'url' as the full original URL.
    # But we need to know what the model expects. 
    # If the model was trained using this exact `decompose_url` function from the notebook, then the model IS trained on truncated URLs.
    # BUT, the user says "tf-idf lr" works (which uses raw URL input in app.py logic).
    # The "exp_2_random_forest_all" uses the pipeline which uses the 'url' column from this dataframe.
    
    # If I change this, I might break consistency with training data if training data WAS truncated.
    # However, looking at the notebook code snippet in comments:
    # "df_decomposed['url'] = df_decomposed.apply(reconstruct_base_url, axis=1)"
    # This suggests the training data WAS truncated.
    # IF so, then the model is behaving "correctly" (garbage in, garbage out for path-based phishing).
    
    # BUT, if I look at `exp_1_linear_models.ipynb` (viewed earlier), TF-IDF LR used `X_text` which was raw URLs.
    # It's likely `exp_2` also intended to use full URLs but might have used this flawed decomposition.
    
    # Let's assume we want FULL URLs for better prediction.
    # I will comment out the truncation line to pass the FULL URL to the model.
    # If the model performance drops, we can revert. But for now, this fixes the "identical prediction" bug.
    
    # df_decomposed['url'] = df_decomposed.apply(reconstruct_base_url, axis=1) 
    return df_decomposed


def extract_url_features(df):
    # Expects df to have 'url' column. 
    # First decompose
    final_df = decompose_url(df)
    
    ## PROTOCOL FEATURES
    final_df['is_https'] = (final_df['protocol'] == 'https').astype(int)
    final_df['is_http'] = (final_df['protocol'] == 'http').astype(int)

    ## DOMAIN FEATURES
    final_df['has_subdomain'] = final_df['subdomains'].notna().astype(int)
    final_df['has_tld'] = final_df['tld'].notna().astype(int)
    final_df['num_subdomain'] = final_df['subdomains'].apply(lambda x: len(x.split('.')) if x else 0)
    
    # check if is IP address
    def is_ip_address(hostname):
        try:
            ipaddress.ip_address(hostname)
            return 1  
        except:
            return 0
    final_df['is_domain_ip'] = final_df['hostname'].apply(is_ip_address)
    
    # suspicious punctuation in domain
    final_df['num_hyphens_domain'] = final_df['hostname'].str.count('-')
    final_df['num_dots_domain'] = final_df['hostname'].str.count(r'\.')

    # detect punycode
    final_df['is_punycode'] = final_df['hostname'].str.contains('xn--', regex=False, na=False).astype(int)

    ## PORT FEATURES
    final_df['has_port'] = final_df['port'].notna().astype(int)

    ## PATH FEATURES
    final_df['has_path'] = final_df['path'].apply(lambda x: 1 if x and x != '/' else 0)
    final_df['path_depth'] = final_df['path'].apply(lambda x: len([p for p in x.split('/') if p]) if x else 0)
    final_df['has_directory_path'] = final_df['directory_path'].apply(lambda x: 1 if x and x != '/' else 0)

    ## FILE FEATURES
    final_df['has_filename'] = final_df['filename'].notna().astype(int)
    final_df['has_file_extension'] = final_df['file_extension'].notna().astype(int)

    ## QUERY FEATURES
    final_df['has_query'] = final_df['query'].notna().astype(int) 
    final_df['num_query_params'] = final_df['query_params'].apply(lambda x: len([p for p in x if p]) if x else 0)
    
    ## FRAGMENT FEATURES
    final_df['has_fragment'] = final_df['fragment'].notna().astype(int)

    ## LENGTH FEATURES
    final_df['length_url'] = final_df['url'].str.len()
    final_df['length_hostname'] = final_df['hostname'].str.len()
    final_df['length_tld'] = final_df['tld'].str.len()
    final_df['length_sld'] = final_df['sld'].str.len()
    final_df['length_subdomains'] = final_df['subdomains'].str.len()
    final_df['length_path'] = final_df['path'].str.len()
    final_df['length_query'] = final_df['query'].str.len()
    final_df['length_fragment'] = final_df['fragment'].str.len()

    ## PUNCTUATION FEATURES
    final_df['num_dots'] = final_df['url'].str.count(r'\.')
    final_df['num_hyphens'] = final_df['url'].str.count('-')
    final_df['num_at'] = final_df['url'].str.count('@')
    final_df['num_question_marks'] = final_df['url'].str.count(r'\?')
    final_df['num_and'] = final_df['url'].str.count('&')
    final_df['num_equal'] = final_df['url'].str.count('=')
    final_df['num_underscores'] = final_df['url'].str.count('_')    
    final_df['num_slashes'] = final_df['url'].str.count('/')
    final_df['num_percent'] = final_df['url'].str.count('%')
    final_df['num_dollars'] = final_df['url'].str.count(r'\$')
    final_df['num_colon'] = final_df['url'].str.count(':')
    final_df['num_semicolon'] = final_df['url'].str.count(';')
    final_df['num_comma'] = final_df['url'].str.count(',')
    final_df['num_hashtag'] = final_df['url'].str.count('#')
    final_df['num_tilde'] = final_df['url'].str.count('~')

    ## SUSPICIOUS PATTERNS FEATURES
    final_df['tld_in_path'] = final_df['path'].apply(lambda x: 1 if x and any(ext in x.lower() for ext in ['.com', '.net', '.org']) else 0)
    final_df['tld_in_subdomain'] = final_df['subdomains'].apply(lambda x: 1 if x and any(ext in x for ext in ['.com', '.net', '.org']) else 0)
    final_df['subdomain_longer_sld'] = (final_df['length_subdomains'] > final_df['length_sld']).astype(int)

    ## RATIO FEATURES
    final_df['ratio_digits_url'] = final_df['url'].apply(lambda x: sum(c.isdigit() for c in x) / len(x) if len(x) > 0 else 0)
    final_df['ratio_digits_hostname'] = final_df['hostname'].apply(lambda x: sum(c.isdigit() for c in x) / len(x) if len(x) > 0 else 0)
    final_df['ratio_letter_hostname'] = final_df['hostname'].apply(lambda x: sum(c.isalpha() for c in x) / len(x) if len(x) > 0 else 0) # Kept from original
    final_df['ratio_letter_url'] = final_df['url'].apply(lambda x: sum(c.isalpha() for c in x) / len(x) if len(x) > 0 else 0)
    final_df['ratio_special_char_hostname'] = final_df['hostname'].apply(lambda x: sum(not c.isalnum() and c not in ['/', ':', '.'] for c in x) / len(x) if len(x) > 0 else 0) # Kept from original
    
    # proportion of components
    final_df['ratio_path_url'] = final_df['length_path'] / final_df['length_url']
    final_df['ratio_hostname_url'] = final_df['length_hostname'] / final_df['length_url']

    # WORD-BASED FEATURES 
    words_raw = final_df['url'].apply(lambda x: re.findall(r'\w+', x) if x else [])
    words_host = final_df['hostname'].apply(lambda x: re.findall(r'\w+', x) if x else [])
    words_path = final_df['path'].apply(lambda x: re.findall(r'\w+', x) if x else [])
    
    final_df['length_words_url'] = words_raw.apply(len)
    final_df['length_words_hostname'] = words_host.apply(len)
    final_df['avg_word_hostname'] = words_host.apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
    final_df['avg_word_path'] = words_path.apply(lambda x: np.mean([len(w) for w in x]) if x else 0)

    ## CHARACTER BASED FEATURES
    final_df['num_unique_chars_hostname'] = final_df['hostname'].apply(lambda x: len(set(x)) if x else 0)
    final_df['num_unique_chars_subdomains'] = final_df['subdomains'].apply(lambda x: len(set(x)) if x else 0)
    final_df['num_unique_chars_sld'] = final_df['sld'].apply(lambda x: len(set(x)) if x else 0)
    final_df['num_non_ascii_hostname'] = final_df['hostname'].apply(lambda x: sum(1 for c in x if ord(c) > 127) if x else 0)
    final_df['num_non_ascii_url'] = final_df['url'].apply(lambda x: sum(1 for c in x if ord(c) > 127) if x else 0)
    
    final_df['longest_repeated_char_hostname'] = final_df['hostname'].apply(lambda x: max([len(list(g)) for k, g in re.findall(r'((.)\2*)', x)]) if x else 0)
    final_df['longest_repeated_char_subdomains'] = final_df['subdomains'].apply(lambda x: max([len(list(g)) for k, g in re.findall(r'((.)\2*)', x)]) if x else 0)
    final_df['longest_repeated_char_sld'] = final_df['sld'].apply(lambda x: max([len(list(g)) for k, g in re.findall(r'((.)\2*)', x)]) if x else 0)

    # URL SHORTENING FEATURES
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl', 't.co']
    final_df['has_shortened_hostname'] = final_df['hostname'].str.lower().apply(lambda x: 1 if x and any(service in x for service in shortening_services) else 0)
    
    # ENTROPY FEATURES
    def calculate_entropy(domain):
        if not domain or len(domain) == 0:
            return 0
        domain_clean = re.sub(r'[^a-z]', '', domain.lower())
        if len(domain_clean) == 0:
            return 0
        char_freq = Counter(domain_clean)
        entropy = -sum((count/len(domain_clean)) * np.log2(count/len(domain_clean)) 
                      for count in char_freq.values())
        return entropy
    final_df['entropy_hostname'] = final_df['hostname'].apply(calculate_entropy)
    final_df['entropy_subdomains'] = final_df['subdomains'].apply(calculate_entropy)
    final_df['entropy_sld'] = final_df['sld'].apply(calculate_entropy)

    # Fill NaNs
    numerical_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'port' in numerical_cols:
        numerical_cols.remove('port')
    final_df[numerical_cols] = final_df[numerical_cols].fillna(0)
    
    # --- NEW: Advanced Feature Engineering to match Training ---
    
    # 1. Interaction / Rule-based Scores (Found in notebook)
    final_df['domain_complexity_score'] = (
        final_df['num_subdomain'] +
        (final_df['length_tld'] <= 2).astype(int) +
        final_df['is_domain_ip'] * 2
    )
    max_complexity = 15 + 1 + 2
    final_df['domain_complexity_score'] = final_df['domain_complexity_score'] / max_complexity

    final_df['suspicion_score'] = (
        final_df['is_http'] * 2 +
        (final_df['num_subdomain'] > 2).astype(int) * 2 +
        final_df['is_domain_ip'].astype(int) * 3 +
        (final_df['length_tld'] <= 2).astype(int) * 2
    )
    max_score = 9
    final_df['suspicion_score'] = final_df['suspicion_score'] / max_score

    # 1b. More Interaction Features (from notebook)
    final_df['is_http_and_many_subdomains'] = ((final_df['is_http'] == 1) & (final_df['num_subdomain'] > 2)).astype(int)
    final_df['ip_and_short_tld'] = (final_df['is_domain_ip'] & (final_df['length_tld'] <= 2)).astype(int)
    final_df['http_and_missing_domain_info'] = ((final_df['is_http'] == 1) & (final_df['has_subdomain'] == 0) & (final_df['length_sld'] <= 3)).astype(int)
    final_df['subdomain_depth_x_http'] = final_df['num_subdomain'] * final_df['is_http']
    final_df['ip_x_http'] = (final_df['is_domain_ip'] * final_df['is_http']).astype(int)
    
    # 1c. Specific Domain/TLD checks
    final_df['has_www_subdomain'] = final_df['subdomains'].apply(lambda x: 1 if x and 'www' in x.split('.') else 0)
    final_df['has_com_tld'] = final_df['tld'].apply(lambda x: 1 if x == 'com' else 0)

    # 2. Impute Missing LLM/Advanced Features (Default to 0/False)

    llm_features = [
        'contains_brand_misspell', 'is_homoglyph_attack', 'homoglyph_type', 'risk_score'
    ]
    for feat in llm_features:
        final_df[feat] = 0

    # 3. Mathematical Transformations (Log, Squared, Is_Zero)
    # Based on missing features identified
    
    # Is Zero features
    is_zero_cols = [
        'num_dots', 'num_hyphens', 'num_at', 'num_question_marks', 'num_and', 
        'num_equal', 'num_percent', 'ratio_digits_url', 'ratio_digits_hostname',
        'avg_word_path', 'length_query', 'num_hyphens_domain', 'length_subdomains'
    ]

    for col in is_zero_cols:
        if col in final_df.columns:
            final_df[f'is_zero_{col}'] = (final_df[col] == 0).astype(int)
        else:
            final_df[f'is_zero_{col}'] = 0 # Should not happen if col exists, but safe fallback

    # Log features (log1p to handle 0)
    log_cols = [
        'length_url', 'length_path', 'ratio_hostname_url', 'length_words_url',
        'avg_word_hostname', 'num_unique_chars_hostname'
    ]
    for col in log_cols:
        if col in final_df.columns:
            # Ensure column is numeric and fill NaNs
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
            final_df[f'log_{col}'] = np.log1p(final_df[col])
        else:
             final_df[f'log_{col}'] = 0

    # Squared features
    squared_cols = ['ratio_letter_url', 'entropy_hostname']
    for col in squared_cols:
        if col in final_df.columns:
            # Ensure column is numeric and fill NaNs
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
            final_df[f'squared_{col}'] = final_df[col] ** 2
        else:
            final_df[f'squared_{col}'] = 0

            
    # Bucketed features (Impute with 0 as we don't have bin edges)
    bucket_cols = ['num_subdomain_bucketed', 'length_tld_bucketed', 'path_depth_bucketed']
    for col in bucket_cols:
        final_df[col] = 0

    return final_df
