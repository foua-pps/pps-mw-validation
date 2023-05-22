from pathlib import Path

from pps_mw_validation import cloudnet


def test_get_file_info():
    """Test get file info."""
    cloudnet_file = Path("20230415_juelich_iwc-Z-T-method.nc")
    assert (
        cloudnet.get_file_info(cloudnet_file)
        == {'date': '20230415', 'location': 'juelich'}
    )


def test_match_files():
    """Test match files."""
    iwc_files = [
        Path("20230415_juelich_iwc-Z-T-method.nc"),
        Path("20230416_juelich_iwc-Z-T-method.nc"),
        Path("20230415_lindenberg_iwc-Z-T-method.nc"),
    ]
    nwp_files = [
        Path("20230415_lindenberg_ecmwf.nc"),
        Path("20230415_juelich_ecmwf.nc"),
        Path("20230417_juelich_ecmwf.nc"),
        Path("20230416_juelich_ecmwf.nc"),
    ]
    assert cloudnet.match_files(iwc_files, nwp_files) == [
        (
            Path('20230415_juelich_iwc-Z-T-method.nc'),
            Path('20230415_juelich_ecmwf.nc'),
        ),
        (
            Path('20230416_juelich_iwc-Z-T-method.nc'),
            Path('20230416_juelich_ecmwf.nc'),
        ),
        (
            Path('20230415_lindenberg_iwc-Z-T-method.nc'),
            Path('20230415_lindenberg_ecmwf.nc'),
        ),
    ]
