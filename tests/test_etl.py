"""Tests for the listing ETL and data helpers, using tiny synthetic frames."""
import gzip

import pandas as pd
import pytest

from src.listing_etl_pipeline import process_full_listings
from src.neighbourhood_etl_pipeline import normalize
from src.service.DataService import DataService

DROPPED_COLUMNS = [
    'host_id', 'name', 'host_name', 'last_review', 'listing_url',
    'neighborhood_overview', 'notes', 'transit', 'scrape_id', 'last_scraped',
    'summary', 'description', 'access', 'interaction', 'house_rules',
    'picture_url', 'host_url', 'host_since', 'host_about',
    'host_thumbnail_url', 'host_picture_url', 'host_verifications',
    'amenities', 'calendar_updated', 'calendar_last_scraped', 'first_review',
    'host_acceptance_rate', 'experiences_offered', 'thumbnail_url',
    'medium_url', 'xl_picture_url', 'jurisdiction_names', 'country_code',
    'country', 'smart_location', 'require_guest_profile_picture',
    'requires_license', 'require_guest_phone_verification', 'market',
    'cancellation_policy', 'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms',
    'calculated_host_listings_count_shared_rooms', 'license',
    'instant_bookable', 'is_business_travel_ready', 'space',
    'is_location_exact',
]


def _raw_listings() -> pd.DataFrame:
    """Six synthetic listings; the last one has an extreme price (outlier)."""
    n = 6
    data = {col: ['x'] * n for col in DROPPED_COLUMNS}
    data.update({
        'id': list(range(n)),
        'price': [50.0, 52.0, 55.0, 58.0, 60.0, 5000.0],
        'reviews_per_month': [1.0, None, 2.0, 1.5, 0.5, 1.0],
        'weekly_price': ['$300.00'] * n,
        'monthly_price': ['$1,200.00'] * n,
        'security_deposit': ['$100.00'] * n,
        'cleaning_fee': [None, '$20.00', '$20.00', '$20.00', '$20.00', '$20.00'],
        'extra_people': ['$0.00'] * n,
        'host_response_rate': ['90%', '100%', None, '80%', '95%', '70%'],
        'host_is_superhost': ['t', 'f', 't', 'f', 't', 'f'],
        'host_has_profile_pic': ['t'] * n,
        'host_identity_verified': ['f'] * n,
        'has_availability': ['t'] * n,
        'host_response_time': ['within an hour', 'within a day'] * 3,
        'host_neighbourhood': ['Centro'] * n,
        'neighbourhood_group': ['Centro'] * n,
        'neighbourhood': ['Sol', 'Sol', 'Embajadores', 'Sol', 'Embajadores', 'Sol'],
        'room_type': ['Private room', 'Entire home/apt'] * 3,
    })
    return pd.DataFrame(data)


@pytest.fixture
def project_tree(tmp_path, monkeypatch):
    """Create <root>/data/interim and run from <root>/a/b so parents[1] is root."""
    root = tmp_path / 'root'
    (root / 'data' / 'interim').mkdir(parents=True)
    (root / 'data' / 'processed').mkdir(parents=True)
    cwd = root / 'a' / 'b'
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    return root


def test_process_full_listings_cleans_and_caches(project_tree):
    _raw_listings().to_csv(
        project_tree / 'data' / 'interim' / 'listings.csv.gz',
        index=False, compression='gzip')

    df = process_full_listings(cached=False, parent=1)

    # the outlier price is removed and the unused columns are dropped
    assert len(df) == 5
    assert df['price'].max() < 1000
    assert not set(DROPPED_COLUMNS) & set(df.columns)
    # missing values are filled and strings converted to numbers
    assert df['reviews_per_month'].notna().all()
    assert df['cleaning_fee'].tolist() == [0.0, 20.0, 20.0, 20.0, 20.0]
    assert df['host_response_rate'].dtype == float
    # t/f flags become booleans, the response time becomes an integer code
    assert df['host_is_superhost'].dtype == bool
    assert pd.api.types.is_integer_dtype(df['host_response_time'])

    # a second call with cached=True reads the stored file back
    cached = process_full_listings(cached=True, parent=1)
    assert len(cached) == len(df)


def test_process_full_listings_cached_without_file_raises(project_tree):
    with pytest.raises(ValueError, match='not found'):
        process_full_listings(cached=True, parent=1)


def test_merge_dataset_drops_duplicate_columns():
    left = pd.DataFrame({'id': [1, 2], 'a': [10, 20], 'shared': ['l', 'l']})
    right = pd.DataFrame({'id': [1, 2], 'b': [3, 4], 'shared': ['r', 'r']})
    merged = DataService.merge_dataset(left, right, on='id')
    assert list(merged.columns) == ['id', 'a', 'shared', 'b']
    assert merged['shared'].tolist() == ['l', 'l']


def test_convert_object_to_string():
    df = pd.DataFrame({'txt': ['a', 'b'], 'num': [1, 2]})
    out = DataService.convert_object_to_string(df)
    assert isinstance(out['txt'].dtype, pd.StringDtype)
    assert out['num'].dtype == 'int64'


def test_normalize_scales_to_unit_range():
    df = pd.DataFrame({'price': [10.0, 20.0, 30.0]})
    out = normalize(df, 'price')
    assert out['price'].tolist() == [0.0, 0.5, 1.0]
