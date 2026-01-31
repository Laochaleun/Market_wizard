#!/usr/bin/env python3
"""Diagnose GUS API connection and refresh cache."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.persona_manager import GUSClient
from app.config import get_settings

def main():
    settings = get_settings()
    print("=== GUS API Diagnostyka ===")
    print()
    print(f"GUS API URL: {settings.gus_api_base_url}")
    print(f"GUS use live: {settings.gus_use_live}")
    print(f"GUS API key: {'USTAWIONY' if settings.gus_api_key else 'NIE USTAWIONY'}")
    print(f"Cache TTL: {settings.gus_cache_ttl_hours} godzin")
    print()
    
    client = GUSClient()
    
    print("Sprawdzanie cache...")
    print(f"Cache path: {client._cache_path}")
    print(f"Cache valid: {client._cache_valid()}")
    print()
    
    if client._load_cache():
        print("Dane z cache:")
        print(f"  Age groups: {len(client._live_age_groups)} grup")
        print(f"  Gender distribution: {client._live_gender_distribution}")
        print(f"  Location distribution: {client._live_location_distribution}")
        print(f"  Income mean: {client._live_income_mean}")
    else:
        print("Cache pusty lub nieważny")
    
    print()
    print("Probuje pobrać dane z GUS API...")
    try:
        client._fetch_live_data()
        client._save_cache()
        print("Dane pobrane i zapisane do cache")
        print(f"  Age groups: {len(client._live_age_groups)} grup")
        print(f"  Gender distribution: {client._live_gender_distribution}")
        print(f"  Location distribution: {client._live_location_distribution}")
        print(f"  Income mean: {client._live_income_mean}")
    except Exception as e:
        print(f"Blad podczas pobierania danych: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
