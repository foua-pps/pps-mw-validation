=================
pps-mw-validation
=================
--------------------------------------
 a tool for validating pps-mw products
--------------------------------------

This is *pps-mw-validation* Python package toolbox. The
*pps-mw-validation* package is supposed to contain tools
that are useful for validating the *pps-mw* products that
the NWCSAF_ will generate based on data from the passive
microwave sensors of the EUMETSAT Polar System-Second
Generation or EPS-SG_ mission.

The set of *pps-mw* products will include cloud and
precipitation products, e.g. ice water path (IWP) from the
EPS-SG Ice cloud Imager (ICI). ICI is a conical scanner
with a footprint size of about 16 km and with an incidence
angle of about 51°. Reference data with this type of
characteristics are not directly available and the
*pps-mw-validation* package contains tool that can be used
to "resample" reference data to make the data comparable to
that of ICI. 

The *pps-mw-validation* package is under development, actual
EPS-SG data are not yet available, and the package currently
only handles the manipulation and comparisons of test and
reference data.

Reference data
..............

The reference data products handled by the *pps-mw-validation*
package includes:

  * `ACTRIS Cloudnet`_ data: ground-based cloud profiling radar
    data from 13 European sites,

  * DARDAR_ data: cloud properties retrieved from combining the
    CloudSat radar and the CALIPSO lidar measurments,

  * `PPS CMIC`_ data: cloud microphysics data derived from AVHRR
    (Metop), VIIRS (JPSS), and MODIS (EOS),

  * `MWI-ICI L2`_ test data: ice water path data retrieved from
    simulated ICI level1b data.


Comparison methods
..................

Two type of comparison/validation methods are handled:

  * direct comparison to resampled `ACTRIS Cloudnet`_ data

  * statistical comparison to resampled DARDAR_ data for a number
    of predefined region of interests:
    
    * arctic
    * central_antarctica
    * mid_latitude_north
    * mid_latitude_south
    * southern_ocean
    * tropics

.. _NWCSAF: https://www.nwcsaf.org/
.. _EPS-SG: https://www.eumetsat.int/metop-sg
.. _ACTRIS Cloudnet: https://cloudnet.fmi.fi/
.. _DARDAR: https://www.icare.univ-lille.fr/dardar/
.. _PPS CMIC: http://nwcsaf.smhi.se/
.. _MWI-ICI L2: https://www.eumetsat.int/new-version-eps-sg-mwi-ici-l2-test-data


Quickstart
==========


Installation
------------

.. code-block:: console

  $ mamba env create -f pps_mw_validation_environment.yml

  $ conda activate pps-mw-validation

  $ pip install .


Run tests
---------

The package contains a test suite can be run by tox_:

.. code-block:: console 

  $ pip install tox

  $ tox

.. _tox: https://pypi.org/project/tox/

Run package scripts
-------------------

The package currently contains three scripts:

  * resample: resampling of *ACTRIS Cloudnet* and *DARDAR* data,
  * collect: collect data around *ACTRIS Cloudnet* sites or within
    region of interests
  * compare: compare two datasets to each other

that can be used to perform a dataset comparison/validation.
The usage of these scripts are described later in this section,
but the workflow will depend a bit on what type of comparison type
that is of interset, and these workflows are first described.

Direct comparison of CMIC or ICI to ACTRIS Cloudnet IWP data 
............................................................

Workflow:

  1. resample: *ACTRIS Cloudnet* data
  2. collect: *CMIC* or *ICI* data around *ACTRIS Cloudnet* sites
  3. compare: run the *validate-by-site* method

Statistical comparison of CMIC or ICI to DARDAR IWP data
........................................................

Workflow:

  1. resample: *DARDAR* data
  2. collect: *CMIC* or *ICI* and *DARDAR* data within region of interest
  3. compare: run the *validate-by-region* method

The *resample* and *collect* scripts will save the processed
data into new data files.

resample
........

.. code-block:: console

  resample --help
  usage: resample [-h] {cloudnet,dardar} ...

  Run the ppsmw data resampler app.

  positional arguments:
    {cloudnet,dardar}
      cloudnet         Resample CLOUDNET data as observed by a conical scanner.
      dardar           Resample DARDAR data as observed by a conical scanner.

  optional arguments:
    -h, --help         show this help message and exit

collect
.......

.. code-block:: console

  collect --help
  usage: collect [-h] {site,roi} ...

  Run the ppsmw validation data collection app.

  positional arguments:
    {site,roi}
      site      Extract CMIC or ICI data around given Cloudnet radar station.
      roi       Extract CMIC, DARDAR, or ICI stats within given region of interest.

  optional arguments:
    -h, --help  show this help message and exit

compare
.......

.. code-block:: console

  compare --help
  usage: compare [-h] {validate-by-region,cloudnet-distribution,time-series,validate-by-site} ...

  Run the ppsmw data comparison app.

  positional arguments:
    {validate-by-region,cloudnet-distribution,time-series,validate-by-site}
      validate-by-region  Compare CMIC or ICI data to DARDAR IWP distributions.
      cloudnet-distribution
                          Show CLOUDNET IWP distribution.
      time-series         Show time series of CMIC or ICI and CLOUDNET IWP data..
      validate-by-site    Compare CMIC or ICI to CLOUDNET IWP data.

  options:
    -h, --help            show this help message and exit

