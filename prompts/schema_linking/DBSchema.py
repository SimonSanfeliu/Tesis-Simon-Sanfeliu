### ALeRCE Database Schema
# This file contains the schema of the database, with the tables and columns that are used in the database.
# It containts different versions of the schema, with and without descriptions of the columns, and with and without indexes, keys, and relations.
# It also contains the features, with and without descriptions, that are used in the database.
# The different contents are grouped in dictionaries, with the name of the table as the key and the schema as the value, so it is easy to access the schema of a specific table.
###

## DB tables names
db_tables = ["object", "probability", "feature", "magstat", "non_detection", "detection", "step", "taxonomy", "feature_version", "xmatch", "allwise", "dataquality", "gaia_ztf", "ss_ztf", "ps1_ztf", "reference", "pipeline", "information_schema", "forced_photometry"]

## Table Schema in SQL create table format
### DataBase Schema without description
#### Important Tables
context_objectTable='''
CREATE TABLE object (
    oid VARCHAR PRIMARY KEY,
    ndethist INTEGER,
    ncovhist INTEGER,
    mjdstarthist DOUBLE PRECISION,
    mjdendhist DOUBLE PRECISION,
    corrected BOOLEAN,
    stellar BOOLEAN,
    ndet INTEGER,
    g_r_max DOUBLE PRECISION,
    g_r_max_corr DOUBLE PRECISION,
    g_r_mean DOUBLE PRECISION,
    g_r_mean_corr DOUBLE PRECISION,
    meanra DOUBLE PRECISION,
    meandec DOUBLE PRECISION,
    sigmara DOUBLE PRECISION,
    sigmadec DOUBLE PRECISION,
    deltajd DOUBLE PRECISION,
    firstmjd DOUBLE PRECISION,
    lastmjd DOUBLE PRECISION,
    step_id_corr VARCHAR,
    diffpos BOOLEAN,
    reference_change BOOLEAN
);
'''

context_probTable='''
CREATE TABLE probability (
    oid VARCHAR REFERENCES object(oid),
    class_name VARCHAR,
    classifier_name VARCHAR,
    classifier_version VARCHAR,
    probability DOUBLE PRECISION NOT NULL,
    ranking INTEGER NOT NULL,
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);
'''

context_featureTable='''
CREATE TABLE feature (
    oid VARCHAR REFERENCES object(oid),
    name VARCHAR NOT NULL,
    value DOUBLE PRECISION,
    fid INTEGER NOT NULL,
    version VARCHAR REFERENCES feature_version(version) NOT NULL,
    PRIMARY KEY (oid, name, fid, version)
);
'''

context_magstatTable='''
CREATE TABLE magstat (
    oid VARCHAR REFERENCES object(oid),
    fid INTEGER NOT NULL,
    stellar BOOLEAN NOT NULL,
    corrected BOOLEAN NOT NULL,
    ndet INTEGER NOT NULL,
    ndubious INTEGER NOT NULL,
    dmdt_first DOUBLE PRECISION,
    dm_first DOUBLE PRECISION,
    sigmadm_first DOUBLE PRECISION,
    dt_first DOUBLE PRECISION,
    magmean DOUBLE PRECISION,
    magmedian DOUBLE PRECISION,
    magmax DOUBLE PRECISION,
    magmin DOUBLE PRECISION,
    magsigma DOUBLE PRECISION,
    maglast DOUBLE PRECISION,
    magfirst DOUBLE PRECISION,
    magmean_corr DOUBLE PRECISION,
    magmedian_corr DOUBLE PRECISION,
    magmax_corr DOUBLE PRECISION,
    magmin_corr DOUBLE PRECISION,
    magsigma_corr DOUBLE PRECISION,
    maglast_corr DOUBLE PRECISION,
    magfirst_corr DOUBLE PRECISION,
    firstmjd DOUBLE PRECISION,
    lastmjd DOUBLE PRECISION,
    step_id_corr VARCHAR NOT NULL,
    saturation_rate DOUBLE PRECISION
);
'''

context_nonDetTable='''
CREATE TABLE non_detection (
    oid VARCHAR REFERENCES object(oid),
    fid INTEGER NOT NULL,
    mjd DOUBLE PRECISION NOT NULL,
    diffmaglim DOUBLE PRECISION,
    PRIMARY KEY (oid, fid, mjd)
);
'''

context_DetTable='''
CREATE TABLE detection (
    candid BIGINT PRIMARY KEY,
    oid VARCHAR REFERENCES object(oid),
    mjd DOUBLE PRECISION NOT NULL,
    fid INTEGER NOT NULL,
    pid FLOAT NOT NULL,
    diffmaglim DOUBLE PRECISION,
    isdiffpos INTEGER NOT NULL,
    nid INTEGER,
    ra DOUBLE PRECISION NOT NULL,
    dec DOUBLE PRECISION NOT NULL,
    magpsf DOUBLE PRECISION NOT NULL,
    sigmapsf DOUBLE PRECISION NOT NULL,
    magap DOUBLE PRECISION,
    sigmagap DOUBLE PRECISION,
    distnr DOUBLE PRECISION,
    rb DOUBLE PRECISION,
    rbversion VARCHAR,
    drb DOUBLE PRECISION,
    drbversion VARCHAR,
    magapbig DOUBLE PRECISION,
    sigmagapbig DOUBLE PRECISION,
    rfid INTEGER,
    magpsf_corr DOUBLE PRECISION,
    sigmapsf_corr DOUBLE PRECISION,
    sigmapsf_corr_ext DOUBLE PRECISION,
    corrected BOOLEAN NOT NULL,
    dubious BOOLEAN NOT NULL,
    parent_candid BIGINT,
    has_stamp BOOLEAN NOT NULL,
    step_id_corr VARCHAR NOT NULL
);
'''

#### Other Tables
context_stepTable='''
CREATE TABLE step (
    step_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    comments VARCHAR NOT NULL,
    date TIMESTAMP NOT NULL
);
'''

context_taxonomyTable='''
CREATE TABLE taxonomy (
    classifier_name VARCHAR PRIMARY KEY,
    classifier_version VARCHAR PRIMARY KEY,
    classes VARCHAR[] NOT NULL
);
'''

context_featureversionTable='''
CREATE TABLE feature_version (
    version VARCHAR PRIMARY KEY,
    step_id_feature VARCHAR REFERENCES step(step_id),
    step_id_preprocess VARCHAR REFERENCES step(step_id)
);
'''

context_xmatchTable='''
CREATE TABLE xmatch (
    oid VARCHAR REFERENCES object(oid),
    catid VARCHAR,
    oid_catalog VARCHAR NOT NULL,
    dist DOUBLE PRECISION NOT NULL,
    class_catalog VARCHAR,
    period DOUBLE PRECISION
);
'''

context_allwiseTable='''
CREATE TABLE allwise (
    oid_catalog VARCHAR PRIMARY KEY,
    ra DOUBLE PRECISION NOT NULL,
    dec DOUBLE PRECISION NOT NULL,
    w1mpro DOUBLE PRECISION,
    w2mpro DOUBLE PRECISION,
    w3mpro DOUBLE PRECISION,
    w4mpro DOUBLE PRECISION,
    w1sigmpro DOUBLE PRECISION,
    w2sigmpro DOUBLE PRECISION,
    w3sigmpro DOUBLE PRECISION,
    w4sigmpro DOUBLE PRECISION,
    j_m_2mass DOUBLE PRECISION,
    h_m_2mass DOUBLE PRECISION,
    k_m_2mass DOUBLE PRECISION,
    j_msig_2mass DOUBLE PRECISION,
    h_msig_2mass DOUBLE PRECISION,
    k_msig_2mass DOUBLE PRECISION
);
'''

context_dataqualityTable='''
CREATE TABLE dataquality (
    candid BIGINT PRIMARY KEY,
    oid VARCHAR NOT NULL,
    fid INTEGER NOT NULL,
    xpos DOUBLE PRECISION,
    ypos DOUBLE PRECISION,
    chipsf DOUBLE PRECISION,
    sky DOUBLE PRECISION,
    fwhm DOUBLE PRECISION,
    classtar DOUBLE PRECISION,
    mindtoedge DOUBLE PRECISION,
    seeratio DOUBLE PRECISION,
    aimage DOUBLE PRECISION,
    bimage DOUBLE PRECISION,
    aimagerat DOUBLE PRECISION,
    bimagerat DOUBLE PRECISION,
    nneg INTEGER,
    nbad INTEGER,
    sumrat DOUBLE PRECISION,
    scorr DOUBLE PRECISION,
    dsnrms DOUBLE PRECISION,
    ssnrms DOUBLE PRECISION,
    magzpsci DOUBLE PRECISION,
    magzpsciunc DOUBLE PRECISION,
    magzpscirms DOUBLE PRECISION,
    nmatches INTEGER,
    clrcoeff DOUBLE PRECISION,
    clrcounc DOUBLE PRECISION,
    zpclrcov DOUBLE PRECISION,
    zpmed DOUBLE PRECISION,
    clrmed DOUBLE PRECISION,
    clrrms DOUBLE PRECISION,
    exptime DOUBLE PRECISION
    FOREIGN KEY (candid, oid) REFERENCES detection(candid, oid)
);
'''

context_gaia_ztfTable='''
CREATE TABLE gaia_ztf (
    oid VARCHAR REFERENCES object(oid),
    candid BIGINT NOT NULL,
    neargaia DOUBLE PRECISION,
    neargaiabright DOUBLE PRECISION,
    maggaia DOUBLE PRECISION,
    maggaiabright DOUBLE PRECISION,
    unique1 BOOLEAN NOT NULL
    PRIMARY KEY (oid)
);
'''

context_ss_ztfTable='''
CREATE TABLE ss_ztf (
    oid VARCHAR REFERENCES object(oid),
    candid BIGINT NOT NULL,
    ssdistnr DOUBLE PRECISION,
    ssmagnr DOUBLE PRECISION,
    ssnamenr VARCHAR
    PRIMARY KEY (oid)
);
'''

context_ps1_ztfTable='''
CREATE TABLE ps1_ztf (
    oid VARCHAR REFERENCES object(oid),
    candid BIGINT NOT NULL,
    objectidps1 DOUBLE PRECISION,
    sgmag1 DOUBLE PRECISION,
    srmag1 DOUBLE PRECISION,
    simag1 DOUBLE PRECISION,
    szmag1 DOUBLE PRECISION,
    sgscore1 DOUBLE PRECISION,
    distpsnr1 DOUBLE PRECISION,
    objectidps2 DOUBLE PRECISION,
    sgmag2 DOUBLE PRECISION,
    srmag2 DOUBLE PRECISION,
    simag2 DOUBLE PRECISION,
    szmag2 DOUBLE PRECISION,
    sgscore2 DOUBLE PRECISION,
    distpsnr2 DOUBLE PRECISION,
    objectidps3 DOUBLE PRECISION,
    sgmag3 DOUBLE PRECISION,
    srmag3 DOUBLE PRECISION,
    simag3 DOUBLE PRECISION,
    szmag3 DOUBLE PRECISION,
    sgscore3 DOUBLE PRECISION,
    distpsnr3 DOUBLE PRECISION,
    nmtchps INTEGER NOT NULL,
    unique1 BOOLEAN NOT NULL,
    unique2 BOOLEAN NOT NULL,
    unique3 BOOLEAN NOT NULL
    PRIMARY KEY (oid, candid)
);
'''

context_refTable='''
CREATE TABLE reference (
    oid VARCHAR REFERENCES object(oid),
    rfid BIGINT,
    candid BIGINT NOT NULL,
    fid INTEGER NOT NULL,
    rcid INTEGER,
    field INTEGER,
    magnr DOUBLE PRECISION,
    sigmagnr DOUBLE PRECISION,
    chinr DOUBLE PRECISION,
    sharpnr DOUBLE PRECISION,
    ranr DOUBLE PRECISION NOT NULL,
    decnr DOUBLE PRECISION NOT NULL,
    mjdstartref DOUBLE PRECISION NOT NULL,
    mjdendref DOUBLE PRECISION NOT NULL,
    nframesref INTEGER NOT NULL
    PRIMARY KEY (oid, rfid)
);
'''

context_pipelineTable='''
CREATE TABLE pipeline (
    pipeline_id VARCHAR PRIMARY KEY,
    step_id_corr VARCHAR,
    step_id_feat VARCHAR,
    step_id_clf VARCHAR,
    step_id_out VARCHAR,
    step_id_stamp VARCHAR,
    date TIMESTAMP NOT NULL
);
'''
# forced_photometry
context_forced_photometryTable = '''
CREATE TABLE forced_photometry ( /* this table contains information about the forced photometry */
    pid BIGINT PRIMARY KEY, 
    oid VARCHAR REFERENCES object(oid), 
    mjd DOUBLE PRECISION NOT NULL, 
    fid INTEGER NOT NULL, 
    ra DOUBLE PRECISION NOT NULL, 
    dec DOUBLE PRECISION NOT NULL, 
    mag DOUBLE PRECISION NOT NULL, 
    e_mag DOUBLE PRECISION NOT NULL, 
    mag_corr DOUBLE PRECISION, 
    e_mag_corr DOUBLE PRECISION, 
    e_mag_corr_ext DOUBLE PRECISION, 
    isdiffpos INTEGER NOT NULL, 
    corrected BOOLEAN NOT NULL, 
    dubious BOOLEAN NOT NULL, 
    parent_candid BIGINT, 
    has_stamp BOOLEAN NOT NULL, 
    field INTEGER, 
    rcid INTEGER, 
    rfid INTEGER, 
    sciinpseeing DOUBLE PRECISION, 
    scibckgnd DOUBLE PRECISION, 
    scisigpix DOUBLE PRECISION, 
    magzpsci DOUBLE PRECISION, 
    magzpsciunc DOUBLE PRECISION, 
    magzpscirms DOUBLE PRECISION, 
    clrcoeff DOUBLE PRECISION, 
    clrcounc DOUBLE PRECISION, 
    exptime DOUBLE PRECISION, 
    adpctdif1 DOUBLE PRECISION, 
    adpctdif2 DOUBLE PRECISION, 
    diffmaglim DOUBLE PRECISION, 
    programid INTEGER, 
    procstatus VARCHAR, 
    distnr DOUBLE PRECISION, 
    ranr DOUBLE PRECISION, 
    decnr DOUBLE PRECISION, 
    magnr DOUBLE PRECISION, 
    sigmagnr DOUBLE PRECISION, 
    chinr DOUBLE PRECISION, 
    sharpnr DOUBLE PRECISION 
);'''


### Table Schema with the description of the columns: context_tableName_desc
# object
context_objectTable_desc = '''
CREATE TABLE object ( /* this is the most important table. It contains the main statistics of an object, independent of time and band */
    oid VARCHAR PRIMARY KEY,  /* object identifier */
    ndethist INTEGER,  /* number of posible detections above 3 sigma */
    ncovhist INTEGER,  /* number of visits */
    mjdstarthist DOUBLE PRECISION,  /* time of first observation even if not detected */
    mjdendhist DOUBLE PRECISION, /* time of last observation even if not detected */
    corrected BOOLEAN, /* whether the object was corrected */
    stellar BOOLEAN, /* whether the object is likely psf shaped */
    ndet INTEGER, /* number of detections */
    g_r_max DOUBLE PRECISION, /* g-r difference color at maximum */
    g_r_max_corr DOUBLE PRECISION, /* g-r color at maximum*/
    g_r_mean DOUBLE PRECISION, /* mean g-r difference color */
    g_r_mean_corr DOUBLE PRECISION, /* mean g-r color */
    meanra DOUBLE PRECISION,  /* mean right ascension */
    meandec DOUBLE PRECISION,  /* mean declination */
    sigmara DOUBLE PRECISION, /* right ascension dispersion */
    sigmadec DOUBLE PRECISION, /* declination dispersion */
    deltajd DOUBLE PRECISION, /* time difference between last and first detection */
    firstmjd DOUBLE PRECISION, /* time of first detection */
    lastmjd DOUBLE PRECISION, /* time of last detection */
    step_id_corr VARCHAR,
    diffpos BOOLEAN, /* whether the first detection was positive or negative */
    reference_change BOOLEAN /* whether the reference image changes */
);'''
context_objectTable_desc_v2 = '''
CREATE TABLE object ( /* this is the most important table. It contains the main statistics of an object, independent of time and band */
    oid VARCHAR PRIMARY KEY,  /* object identifier, from the ZTF */
    deltajd DOUBLE PRECISION, /* time difference between last and first detection */
    firstmjd DOUBLE PRECISION, /* time of first detection */
    lastmjd DOUBLE PRECISION, /* time of last detection */
    ndethist INTEGER,  /* number of posible detections above 3 sigma */
    ncovhist INTEGER,  /* number of visits */
    mjdstarthist DOUBLE PRECISION,  /* time of first observation even if not detected, Earliest Julian date of epoch corresponding to ndethist [days]*/
    mjdendhist DOUBLE PRECISION, /* time of last observation even if not detected, Latest Julian date of epoch corresponding to ndethist [days] */
    corrected BOOLEAN, /* whether the object was corrected */
    stellar BOOLEAN, /* whether the object is likely psf shaped */
    ndet INTEGER, /* total number of detections for the object */
    g_r_max DOUBLE PRECISION, /* g-r difference color at maximum */
    g_r_max_corr DOUBLE PRECISION, /* g-r color at maximum*/
    g_r_mean DOUBLE PRECISION, /* mean g-r difference color */
    g_r_mean_corr DOUBLE PRECISION, /* mean g-r color */
    meanra DOUBLE PRECISION,  /* mean right ascension */
    meandec DOUBLE PRECISION,  /* mean declination */
    sigmara DOUBLE PRECISION, /* right ascension dispersion */
    sigmadec DOUBLE PRECISION, /* declination dispersion */
    step_id_corr VARCHAR, /* correction step pipeline version */
    diffpos BOOLEAN, /* whether the first detection was positive or negative */
    reference_change BOOLEAN /* whether the reference image changes */
);'''
# probability
context_probTable_desc = '''CREATE TABLE probability ( /* this table contains the machine learning derived classification probabilities and rankings */
    oid VARCHAR REFERENCES object(oid),
    class_name VARCHAR, /* name of the class */
    classifier_name VARCHAR, /* name of the classifier */
    classifier_version VARCHAR, /* version of the classiifer */
    probability DOUBLE PRECISION NOT NULL, /* probability of the class given a classifier and version */
    ranking INTEGER NOT NULL, /* class probability ranking (1 is the most likely class) */
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);'''
context_probTable_desc_v2 = '''CREATE TABLE probability ( /* this table contains the machine learning derived classification probabilities and rankings */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    class_name VARCHAR, /* name of the class */
    classifier_name VARCHAR, /* name of the classifier */
    classifier_version VARCHAR, /* version of the classiifer */
    probability DOUBLE PRECISION NOT NULL, /* probability of the class given a classifier and version */
    ranking INTEGER NOT NULL, /* class probability ranking (1 is the most likely class) */
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);'''
# magstat
context_magstatTable_desc='''CREATE TABLE magstat ( /* different statistics for the object divided by band or filter */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    fid INTEGER NOT NULL, /* band or filter identifier (1=g; 2=r; 3=i) */
    stellar BOOLEAN NOT NULL, /* whether we believe the object is stellar */
    corrected BOOLEAN NOT NULL, /* whether the object’s light curve has been corrected */
    ndet INTEGER NOT NULL, /* the object number of detection in the given band */
    ndubious INTEGER NOT NULL, /* the points in the light curve in the given band that we don’t trust  */
    dmdt_first DOUBLE PRECISION, /* lower limit for the the rate of magnitude change at detection in the given band */
    dm_first DOUBLE PRECISION, /* change in magnitude with respect to the last non detection at detection in the given band */
    sigmadm_first DOUBLE PRECISION, /* error in the change of magnitude w.r.t. the last detection in the given band */
    dt_first DOUBLE PRECISION, /* time between the last non detection and the first detection for the given band */
    magmean DOUBLE PRECISION, /* mean difference magnitude for the given band */
    magmedian DOUBLE PRECISION, /* median difference magnitude for the given band */
    magmax DOUBLE PRECISION, /* maximum difference magnitude for the given band */
    magmin DOUBLE PRECISION, /* minimum difference magnitude for the given band */
    magsigma DOUBLE PRECISION, /* dispersion of the difference magnitude for the given band */
    maglast DOUBLE PRECISION, /* last difference magnitude for the given band */
    magfirst DOUBLE PRECISION, /* first difference magnitude for the given band */
    magmean_corr DOUBLE PRECISION, /* mean apparent (corrected) magnitude for the given band */
    magmedian_corr DOUBLE PRECISION, /* median apparent (corrected) magnitude for the given band */
    magmax_corr DOUBLE PRECISION, /* maximum apparent (corrected) magnitude for the given band */
    magmin_corr DOUBLE PRECISION, /* minimum apparent (corrected) magnitude for the given band */
    magsigma_corr DOUBLE PRECISION, /* dispersion of the apparent (corrected) magnitude for the given band */
    maglast_corr DOUBLE PRECISION, /* last apparent (corrected) magnitude for the given band */
    magfirst_corr DOUBLE PRECISION, /* mean apparent (corrected) magnitude for the given band */
    firstmjd DOUBLE PRECISION, /* time of first detection for the given band */
    lastmjd DOUBLE PRECISION, /* time of last detection for the given band */
    step_id_corr VARCHAR NOT NULL, /* correction step id */
    saturation_rate DOUBLE PRECISION /* saturation level for the given band */
);'''
# feature
context_featureTable_desc='''CREATE TABLE feature ( /* table with features for a given object, each row contains one feature per object */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    name VARCHAR NOT NULL, /* name of the feature */
    value DOUBLE PRECISION, /* value of the feature */
    fid INTEGER NOT NULL, /* which bandpass this is based (1: g, 2: r, 3: i, 12: gr, 13: gi, 23: ri, 123: gri, ) */
    version VARCHAR REFERENCES feature_version(version) NOT NULL, /* version of the features (see table feature_version) */
    PRIMARY KEY (oid, name, fid, version)
);'''
# ss_ztf
context_ss_ztfTable_desc='''CREATE TABLE ss_ztf ( /* this table contains information about the closest known solar system object */
    oid VARCHAR REFERENCES object(oid),
    candid BIGINT NOT NULL, /* unique candidate identifier */
    ssdistnr DOUBLE PRECISION, /* distance to nearest known solar system object */
    ssmagnr DOUBLE PRECISION, /* magnitude of nearest known solar system object */
    ssnamenr VARCHAR /* name to nearest known solar system object. If the object is not a known solar system object, this field is 'null' (VARCHAR) */
    PRIMARY KEY (oid)
);'''
# detection
context_DetTable_desc='''CREATE TABLE detection (  /* this table contains information about the object detections, or its light curve. Avoid doing filters on this table in any parameter other than oid */
    candid BIGINT PRIMARY KEY, /* unique candidate identifier */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    mjd DOUBLE PRECISION NOT NULL, /* time of detection in modified julian date */
    fid INTEGER NOT NULL, /* filter identifier  (1=g; 2=r; 3=i)*/
    pid FLOAT NOT NULL, /* program identifier */
    diffmaglim DOUBLE PRECISION, /* limiting difference magnitud */
    isdiffpos INTEGER NOT NULL, /* whether the difference is positive or negative */
    nid INTEGER, /* unique night identifier */
    ra DOUBLE PRECISION NOT NULL, /* inferred right ascension */
    dec DOUBLE PRECISION NOT NULL, /* inferred declination */
    magpsf DOUBLE PRECISION NOT NULL, /* point spread function (psf) difference magnitude */
    sigmapsf DOUBLE PRECISION NOT NULL, /* psf difference magnitude error */
    magap DOUBLE PRECISION, /* aperture difference magnitude */
    sigmagap DOUBLE PRECISION, /* aperture difference magnitud error */
    distnr DOUBLE PRECISION, /* distance to the nearest source in the reference image */
    rb DOUBLE PRECISION, /* ZTF real bogus score */
    rbversion VARCHAR, /* version of the ZTF real bogus score */
    drb DOUBLE PRECISION, /* ZTF deep learning based real bogus score */
    drbversion VARCHAR, /* versio  of the ZTF deep learning based real bogus score */
    magapbig DOUBLE PRECISION, /* large aperture magnitude */
    sigmagapbig DOUBLE PRECISION, /* large aperture magnitude error */
    rfid INTEGER, /* identifier of the reference image used for the difference image */
    magpsf_corr DOUBLE PRECISION, /* apparent magnitude (corrected difference magnitude) */
    sigmapsf_corr DOUBLE PRECISION, /* error of the apparent magnitude assuming point like source */
    sigmapsf_corr_ext DOUBLE PRECISION, /* error of the apparent magnitude assuming extended source */
    corrected BOOLEAN NOT NULL, /* whether the object’s magnitude was corrected */
    dubious BOOLEAN NOT NULL, /* whether the object is dubious or not */
    parent_candid BIGINT, /* identifier of the candidate where this information was generated (this happens if the given detection does not pass the real bogus filter, but a later detection does */
    has_stamp BOOLEAN NOT NULL, /* whether the candidate has an associated image stamp (same as before */
    step_id_corr VARCHAR NOT NULL /* identifier of the correction step */
);'''
# ps1_ztf
context_ps1_ztfTable_desc = '''CREATE TABLE ps1_ztf ( /* information about the three closest sources in Pan STARRS */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    candid BIGINT NOT NULL, /* unique candidate identifier */
    objectidps1 DOUBLE PRECISION, /* identifier of the closest source in pan starrs (ps) */
    sgmag1 DOUBLE PRECISION, /* ps g band magnitude */
    srmag1 DOUBLE PRECISION, /* ps r band magnitude */
    simag1 DOUBLE PRECISION, /* ps i band magnitude */
    szmag1 DOUBLE PRECISION, /* ps z band magnitude */
    sgscore1 DOUBLE PRECISION, /* ps star galaxy score */
    distpsnr1 DOUBLE PRECISION, /* distance to closest source in panstarrs */
    objectidps2 DOUBLE PRECISION,  /* identifier of the second closest source in pan starrs (ps) */
    sgmag2 DOUBLE PRECISION, /* g-band PSF magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag] */
    srmag2 DOUBLE PRECISION, /* r-band PSF magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag] */
    simag2 DOUBLE PRECISION, /* i-band PSF magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag] */
    szmag2 DOUBLE PRECISION, /* z-band PSF magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag] */
    sgscore2 DOUBLE PRECISION, /* Star/Galaxy score of second closest source from PS1 catalog; if exists within 30 arcsec: 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star */
    distpsnr2 DOUBLE PRECISION, /* Distance to second closest source from PS1 catalog; if exists within 30 arcsec [arcsec] */
    objectidps3 DOUBLE PRECISION, /* Object ID of third closest source from PS1 catalog; if exists within 30 arcsec */
    sgmag3 DOUBLE PRECISION, /* g-band PSF magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag] */
    srmag3 DOUBLE PRECISION, /* r-band PSF magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag] */
    simag3 DOUBLE PRECISION, /* i-band PSF magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag] */
    szmag3 DOUBLE PRECISION, /* z-band PSF magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag] */
    sgscore3 DOUBLE PRECISION, /* Star/Galaxy score of third closest source from PS1 catalog; if exists within 30 arcsec: 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star */
    distpsnr3 DOUBLE PRECISION, /* Distance to third closest source from PS1 catalog; if exists within 30 arcsec [arcsec] */
    nmtchps INTEGER NOT NULL, /* Number of source matches from PS1 catalog falling within 30 arcsec */
    unique1 BOOLEAN NOT NULL, /* Whether the first closest object has changed w.r.t the first alert */
    unique2 BOOLEAN NOT NULL, /* Whether the second closest object has changed w.r.t the first alert */
    unique3 BOOLEAN NOT NULL /* Whether the third closest object has changed w.r.t the first alert */
    PRIMARY KEY (oid, candid)
);'''
# forced_photometry
context_forced_photometryTable_desc = '''CREATE TABLE forced_photometry ( /* this table contains information about the forced photometry */
    pid BIGINT PRIMARY KEY, /* unique candidate identifier */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    mjd DOUBLE PRECISION NOT NULL, /* time of detection in modified julian date */
    fid INTEGER NOT NULL, /* filter identifier  (1=g; 2=r; 3=i)*/
    ra DOUBLE PRECISION NOT NULL, /* inferred right ascension */
    dec DOUBLE PRECISION NOT NULL, /* inferred declination */
    mag DOUBLE PRECISION NOT NULL, /* point spread function (psf) difference magnitude */
    e_mag DOUBLE PRECISION NOT NULL, /* psf difference magnitude error */
    mag_corr DOUBLE PRECISION, /* apparent magnitude (corrected difference magnitude) */
    e_mag_corr DOUBLE PRECISION, /* error of the apparent magnitude assuming point like source */
    e_mag_corr_ext DOUBLE PRECISION, /* error of the apparent magnitude assuming extended source */
    isdiffpos INTEGER NOT NULL, /* whether the difference is positive or negative */
    corrected BOOLEAN NOT NULL, /* whether the object’s magnitude was corrected */
    dubious BOOLEAN NOT NULL, /* whether the object is dubious or not */
    parent_candid BIGINT, /* identifier of the candidate where this information was generated (this happens if the given detection does not pass the real bogus filter, but a later detection does */
    has_stamp BOOLEAN NOT NULL, /* whether the candidate has an associated image stamp (same as before */
    field INTEGER, /* field identifier */
    rcid INTEGER, /* reference catalog identifier */
    rfid INTEGER, /* identifier of the reference image used for the difference image */
    sciinpseeing DOUBLE PRECISION, /* seeing in the science image */
    scibckgnd DOUBLE PRECISION, /* background in the science image */
    scisigpix DOUBLE PRECISION, /* sigma per pixel in the science image */
    magzpsci DOUBLE PRECISION, /* zero point magnitude in the science image */
    magzpsciunc DOUBLE PRECISION, /* uncertainty in the zero point magnitude in the science image */
    magzpscirms DOUBLE PRECISION, /* rms of the zero point magnitude in the science image */
    clrcoeff DOUBLE PRECISION, /* color correction coefficient */
    clrcounc DOUBLE PRECISION, /* uncertainty in the color correction coefficient */
    exptime DOUBLE PRECISION, /* exposure time */
    adpctdif1 DOUBLE PRECISION, /* percentage difference in aperture photometry */
    adpctdif2 DOUBLE PRECISION, /* percentage difference in aperture photometry */
    diffmaglim DOUBLE PRECISION, /* limiting difference magnitude */
    programid INTEGER, /* program identifier */
    procstatus VARCHAR, /* processing status */
    distnr DOUBLE PRECISION, /* distance to the nearest source in the reference image */
    ranr DOUBLE PRECISION, /* right ascension of the nearest source in the reference image */
    decnr DOUBLE PRECISION, /* declination of the nearest source in the reference image */
    magnr DOUBLE PRECISION, /* magnitude of the nearest source in the reference image */
    sigmagnr DOUBLE PRECISION, /* magnitude error of the nearest source in the reference image */
    chinr DOUBLE PRECISION, /* chi of the nearest source in the reference image */
    sharpnr DOUBLE PRECISION /* sharpness of the nearest source in the reference image */
);'''
# non_detection
context_nonDetTable_desc='''
CREATE TABLE non_detection (
    oid VARCHAR REFERENCES object(oid), /* unique identifier for this object */
    fid INTEGER NOT NULL, /* Filter ID (1=g; 2=r; 3=i) */
    mjd DOUBLE PRECISION NOT NULL, /* Observation Julian date at start of exposure [days] */
    diffmaglim DOUBLE PRECISION, /* 5-sigma mag limit in difference image based on PSF-fit photometry [mag] */
    PRIMARY KEY (oid, fid, mjd)
);
'''
# step
context_stepTable_desc='''
CREATE TABLE step (
    step_id VARCHAR PRIMARY KEY, /* unique identifier of the step */
    name VARCHAR NOT NULL, /* stamp, corr, feat, clf, out */
    version VARCHAR NOT NULL, /* relevant versions as a dictionary */
    comments VARCHAR NOT NULL, /* what this does */
    date TIMESTAMP NOT NULL /* date of docker creation */
);
'''
# taxonomy
context_taxonomyTable_desc='''
CREATE TABLE taxonomy (
    classifier_name VARCHAR PRIMARY KEY, /* name of the taxonomy */
    classifier_version VARCHAR PRIMARY KEY, /* taxonomy version */
    classes VARCHAR[] NOT NULL /* classes in the hierarchical form they are computed */
);
'''
# feature_version
context_featureversionTable_desc='''
CREATE TABLE feature_version (
    version VARCHAR PRIMARY KEY, /* feature processing version */
    step_id_feature VARCHAR REFERENCES step(step_id), /* feture step docker id*/
    step_id_preprocess VARCHAR REFERENCES step(step_id) /* preprocess step docker id*/
);
'''
# xmatch
context_xmatchTable_desc='''
CREATE TABLE xmatch (
    oid VARCHAR REFERENCES object(oid), /* ZTF object Id */
    catid VARCHAR, /* Catalog Id / Name */
    oid_catalog VARCHAR NOT NULL, /* Object Id in Catalog */
    dist DOUBLE PRECISION NOT NULL, /* Distance to the closest ZTF object [arcsec] */
    class_catalog VARCHAR, /* Class in the given gatalog if reported */
    period DOUBLE PRECISION /* Period in the given catalog if reported  [days] */
);
'''
# xmatch
context_xmatchTable_desc_v2='''
CREATE TABLE xmatch (
    oid VARCHAR REFERENCES object(oid), /* ZTF object Id, same 'oid' from the others tables */
    catid VARCHAR, /* Catalog Id / Name from which the object was matched */
    oid_catalog VARCHAR NOT NULL, /* Object Id in Catalog */
    dist DOUBLE PRECISION NOT NULL, /* Distance to the closest ZTF object [arcsec] */
    class_catalog VARCHAR, /* Class in the given gatalog if reported */
    period DOUBLE PRECISION /* Period in the given catalog if reported  [days] */
);
'''
# allwise
context_allwiseTable_desc='''
CREATE TABLE allwise (
    oid_catalog VARCHAR PRIMARY KEY, /* object Id inside catalog, correspond to column: designation  in ALLWISE : Sexagesimal, equatorial position-based source name in the form: hhmmss.ss+ddmmss.s. The full naming convention for AllWISE Source Catalog sources has the form "WISEA  Jhhmmss.ss+ddmmss.s," where "WISEA" indicates the source is from the AllWISE Source Catalog, and "J" indicates the position is J2000. The "WISEA" acronym is not listed explicitly in the designation column. */
    ra DOUBLE PRECISION NOT NULL, /* J2000 right ascension with respect to the 2MASS PSC reference frame from the non-moving source extraction. */
    dec DOUBLE PRECISION NOT NULL, /* J2000 declination with respect to the 2MASS PSC reference frame from the non-moving source extraction. */
    w1mpro DOUBLE PRECISION, /* W1 magnitude measured with profile-fitting photometry, or the magnitude of the 95% confidence brightness upper limit if the W1 flux measurement has SNR<2. This column is null if the source is nominally detected in W1, but no useful brightness estimate could be made. */
    w2mpro DOUBLE PRECISION, /* analogous to w1mpro */
    w3mpro DOUBLE PRECISION, /* analogous to w1mpro */
    w4mpro DOUBLE PRECISION, /* analogous to w1mpro */
    w1sigmpro DOUBLE PRECISION, /* W1 profile-fit photometric measurement uncertainty in mag units. This column is null if the W1 profile-fit magnitude is a 95% confidence upper limit or if the source is not measurable. */
    w2sigmpro DOUBLE PRECISION, /* analogous to w1sigmpro */
    w3sigmpro DOUBLE PRECISION, /* analogous to w1sigmpro */
    w4sigmpro DOUBLE PRECISION, /* analogous to w1sigmpro */
    j_m_2mass DOUBLE PRECISION, /* 2MASS J-band magnitude or magnitude upper limit of the associated 2MASS PSC source. This column is "null" if there is no associated 2MASS PSC source or if the 2MASS PSC J-band magnitude entry is "null". */
    h_m_2mass DOUBLE PRECISION, /* analogous to j_m_2mass */
    k_m_2mass DOUBLE PRECISION, /* analogous to j_m_2mass */
    j_msig_2mass DOUBLE PRECISION, /* 2MASS J-band corrected photometric uncertainty of the associated 2MASS PSC source. This column is "null" if there is no associated 2MASS PSC source or if the 2MASS PSC J-band uncertainty entry is "null". */
    h_msig_2mass DOUBLE PRECISION, /* analogous to j_msig_2mass */
    k_msig_2mass DOUBLE PRECISION /* analogous to j_msig_2mass */
);
'''
# dataquality
context_dataqualityTable_desc='''
CREATE TABLE dataquality (
    candid BIGINT PRIMARY KEY, /* unique identifier for the subtraction candidate */
    oid VARCHAR NOT NULL, /* unique identifier for this object */
    fid INTEGER NOT NULL, /* Filter ID (1=g; 2=r; 3=i) */
    xpos DOUBLE PRECISION, /* x-image position of candidate [pixels] */
    ypos DOUBLE PRECISION, /* y-image position of candidate [pixels] */
    chipsf DOUBLE PRECISION, /* Reduced chi-square for PSF-fit */
    sky DOUBLE PRECISION, /* Local sky background estimate [DN] */
    fwhm DOUBLE PRECISION, /* Full Width Half Max assuming a Gaussian core, from SExtractor [pixels] */
    classtar DOUBLE PRECISION, /* Star/Galaxy classification score from SExtractor */
    mindtoedge DOUBLE PRECISION, /* Distance to nearest edge in image [pixels] */
    seeratio DOUBLE PRECISION, /* Ratio: difffwhm / fwhm */
    aimage DOUBLE PRECISION, /* Windowed profile RMS afloat major axis from SExtractor [pixels] */
    bimage DOUBLE PRECISION, /* Windowed profile RMS afloat minor axis from SExtractor [pixels] */
    aimagerat DOUBLE PRECISION, /* Ratio: aimage / fwhm */
    bimagerat DOUBLE PRECISION, /* Ratio: bimage / fwhm */
    nneg INTEGER, /* number of negative pixels in a 5 x 5 pixel stamp */
    nbad INTEGER, /* number of prior-tagged bad pixels in a 5 x 5 pixel stamp */
    sumrat DOUBLE PRECISION, /* Ratio: sum(pixels) / sum(abs(pixels)) in a 5 x 5 pixel stamp where stamp is first median-filtered to mitigate outliers */
    scorr DOUBLE PRECISION, /* Peak-pixel signal-to-noise ratio in point source matched-filtered detection image */
    dsnrms DOUBLE PRECISION, /* Ratio: D/stddev(D) on event position where D = difference image */
    ssnrms DOUBLE PRECISION, /* Ratio: S/stddev(S) on event position where S = image of convolution: D (x) PSF(D) */
    magzpsci DOUBLE PRECISION, /* Magnitude zero point for photometry estimates [mag] */
    magzpsciunc DOUBLE PRECISION, /* Magnitude zero point uncertainty (in magzpsci) [mag] */
    magzpscirms DOUBLE PRECISION, /* RMS (deviation from average) in all differences between instrumental photometry and matched photometric calibrators from science image processing [mag] */
    nmatches INTEGER, /* Number of PS1 photometric calibrators used to calibrate science image from science image processing */
    clrcoeff DOUBLE PRECISION, /* Color coefficient from linear fit from photometric calibration of science image */
    clrcounc DOUBLE PRECISION, /* Color coefficient uncertainty from linear fit (corresponding to clrcoeff) */
    zpclrcov DOUBLE PRECISION, /* Covariance in magzpsci and clrcoeff from science image processing [mag^2] */
    zpmed DOUBLE PRECISION, /* Magnitude zero point from median of all differences between instrumental photometry and matched photometric calibrators from science image processing [mag] */
    clrmed DOUBLE PRECISION, /* Median color of all PS1 photometric calibrators used from science image processing [mag]: for filter (fid) = 1, 2, 3, PS1 color used = g-r, g-r, r-i respectively */
    clrrms DOUBLE PRECISION, /* RMS color (deviation from average) of all PS1 photometric calibrators used from science image processing [mag] */
    exptime DOUBLE PRECISION /* Integration time of camera exposure [sec] */
    FOREIGN KEY (candid, oid) REFERENCES detection(candid, oid)
);
'''
# gaia_ztf
context_gaia_ztfTable_desc='''
CREATE TABLE gaia_ztf (
    oid VARCHAR REFERENCES object(oid), /* unique identifier for this object */
    candid BIGINT NOT NULL, /* unique identifier for the subtraction candidate */
    neargaia DOUBLE PRECISION, /* Distance to closest source from Gaia DR1 catalog irrespective of magnitude; if exists within 90 arcsec [arcsec] */
    neargaiabright DOUBLE PRECISION, /* Distance to closest source from Gaia DR1 catalog brighter than magnitude 14; if exists within 90 arcsec [arcsec] */
    maggaia DOUBLE PRECISION, /* Gaia (G-band) magnitude of closest source from Gaia DR1 catalog irrespective of magnitude; if exists within 90 arcsec [mag] */
    maggaiabright DOUBLE PRECISION, /* Gaia (G-band) magnitude of closest source from Gaia DR1 catalog brighter than magnitude 14; if exists within 90 arcsec [mag] */
    unique1 BOOLEAN NOT NULL /* whether the closest object has changed w.r.t the first alert */
    PRIMARY KEY (oid)
);
'''
# ref
context_refTable_desc ='''
CREATE TABLE reference (
    oid VARCHAR REFERENCES object(oid), /* unique identifier for this object */
    rfid BIGINT, /* Processing ID for reference image to facilitate archive retrieval */
    candid BIGINT NOT NULL, /* unique identifier for the subtraction candidate (first detection for the template) */
    fid INTEGER NOT NULL, /* Filter ID (1=g; 2=r; 3=i) */
    rcid INTEGER, /* Readout channel ID [00 .. 63] */
    field INTEGER, /* ZTF field ID */
    magnr DOUBLE PRECISION, /* magnitude of nearest source in reference image PSF-catalog within 30 arcsec [mag] */
    sigmagnr DOUBLE PRECISION, /* 1-sigma uncertainty in magnr within 30 arcsec [mag] */
    chinr DOUBLE PRECISION, /* DAOPhot chi parameter of nearest source in reference image PSF-catalog within 30 arcsec */
    sharpnr DOUBLE PRECISION, /* DAOPhot sharp parameter of nearest source in reference image PSF-catalog within 30 arcsec */
    ranr DOUBLE PRECISION NOT NULL, /* Right Ascension of nearest source in reference image PSF-catalog; J2000 [deg] */
    decnr DOUBLE PRECISION NOT NULL, /* Declination of nearest source in reference image PSF-catalog; J2000 [deg] */
    mjdstartref DOUBLE PRECISION NOT NULL, /* Observation Modified Julian date of earliest exposure used to generate reference image [days] */
    mjdendref DOUBLE PRECISION NOT NULL, /* Observation Modified Julian date of latest exposure used to generate reference image [days] */
    nframesref INTEGER NOT NULL /* Number of frames (epochal images) used to generate reference image */
    PRIMARY KEY (oid, rfid)
);
'''
# pipeline
context_pipelineTable_desc='''
CREATE TABLE pipeline (
    pipeline_id VARCHAR PRIMARY KEY, /* Pipeline version */
    step_id_corr VARCHAR, /* step version */
    step_id_feat VARCHAR, /* step version */
    step_id_clf VARCHAR, /* step version */
    step_id_out VARCHAR, /* step version */
    step_id_stamp VARCHAR, /* step version */
    date TIMESTAMP NOT NULL /* date of implementation */
);
'''



### Table Schema, w/ descriptions for all columns in important tables in one string
context_db_d = '''
CREATE TABLE object (
    oid VARCHAR PRIMARY KEY, -- unique identifier for this object
    ndethist INTEGER, -- Number of spatially-coincident detections falling within 1.5 arcsec going back to beginning of survey; only detections that fell on the same field and readout-channel ID where the input candidate was observed are counted. All raw detections down to a photometric S/N of ~ 3 are included.
    ncovhist INTEGER, -- Number of times input candidate position fell on any field and readout-channel going back to beginning of survey
    mjdstarthist DOUBLE PRECISION, -- Earliest Julian date of epoch corresponding to ndethist [days]
    mjdendhist DOUBLE PRECISION, -- Latest Julian date of epoch corresponding to ndethist [days]
    corrected BOOLEAN, -- whether the corrected light curve was computed and can be used
    stellar BOOLEAN, -- whether the object is a likely stellar-like source
    ndet INTEGER, -- total number of detections for the object
    g_r_max DOUBLE PRECISION, -- difference between the minimum g and r band difference magnitudes
    g_r_max_corr DOUBLE PRECISION, -- difference between the minimum g and r band corrected magnitudes
    g_r_mean DOUBLE PRECISION, -- difference between the mean g and r band difference magnitudes
    g_r_mean_corr DOUBLE PRECISION, -- difference between the mean g and r band corrected magnitudes
    meanra DOUBLE PRECISION, -- mean right ascension
    meandec DOUBLE PRECISION, -- mean declination
    sigmara DOUBLE PRECISION, -- right ascension standard deviation
    sigmadec DOUBLE PRECISION, -- declination standard deviation
    deltajd DOUBLE PRECISION, -- difference between last and first detection time
    firstmjd DOUBLE PRECISION, -- first detection time
    lastmjd DOUBLE PRECISION, -- last detection time
    step_id_corr VARCHAR, -- correction step pipeline version
    diffpos BOOLEAN, --
    reference_change BOOLEAN --
);

CREATE TABLE probability (
    oid VARCHAR REFERENCES object(oid), -- unique object identifier
    class_name VARCHAR, -- class name
    classifier_name VARCHAR, -- classifier name
    classifier_version VARCHAR, -- classifier version
    probability DOUBLE PRECISION NOT NULL, -- probability
    ranking INTEGER NOT NULL, -- ranking within the classifier type
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);

CREATE TABLE feature (
    oid VARCHAR REFERENCES object(oid), -- unique object identifier
    name VARCHAR NOT NULL, -- name of the feature
    value DOUBLE PRECISION, -- value of the feature
    fid INTEGER NOT NULL, -- which bandpass this is based (1: g, 2: r, 3: i, 12: gr, 13: gi, 23: ri, 123: gri, )
    version VARCHAR REFERENCES feature_version(version) NOT NULL, -- version of the features (see table feature_version)
    PRIMARY KEY (oid, name, fid, version)
);

CREATE TABLE magstat (
    oid VARCHAR REFERENCES object(oid), -- unique identifier for this object
    fid INTEGER NOT NULL, -- Filter ID (1=g; 2=r; 3=i)
    stellar BOOLEAN NOT NULL, -- whether the object appears to be unresolved in the given band
    corrected BOOLEAN NOT NULL, -- whether the corrected photometry should be used
    ndet INTEGER NOT NULL, -- number of detections in the given band
    ndubious INTEGER NOT NULL, -- number of dubious corrections
    dmdt_first DOUBLE PRECISION, -- initial rise estimate, the maximum slope between the first detection and any previous non-detection
    dm_first DOUBLE PRECISION, -- the magnitude difference used to compute dmdt_first
    sigmadm_first DOUBLE PRECISION, -- the error of the magnitude difference used to compute dmdt_first
    dt_first DOUBLE PRECISION, -- the time difference used to compute dmdt_first
    magmean DOUBLE PRECISION, -- the mean magnitude for the given fid
    magmedian DOUBLE PRECISION, -- the median magnitude for the given fid
    magmax DOUBLE PRECISION, -- the max magnitude for the given fid
    magmin DOUBLE PRECISION, -- the min magnitude for the given fid
    magsigma DOUBLE PRECISION, -- magnitude standard deviation for the given fid
    maglast DOUBLE PRECISION, -- the last magnitude for the given fid
    magfirst DOUBLE PRECISION, -- the first magnitude for the given fid
    magmean_corr DOUBLE PRECISION, -- the mean corrected magnitude for the given fid
    magmedian_corr DOUBLE PRECISION, -- the median corrected magnitude for the given fid
    magmax_corr DOUBLE PRECISION, -- the max corrected magnitude for the given fid
    magmin_corr DOUBLE PRECISION, -- the min corrected magnitude for the given fid
    magsigma_corr DOUBLE PRECISION, -- corrected magnitude standard deviation for the given fid
    maglast_corr DOUBLE PRECISION, -- the last corrected magnitude for the given fid
    magfirst_corr DOUBLE PRECISION, -- the first corrected magnitude for the given fid
    firstmjd DOUBLE PRECISION, -- the time of the first detection in the given fid
    lastmjd DOUBLE PRECISION, -- the time of the last detection in the given fid
    step_id_corr VARCHAR NOT NULL, -- correction step pipeline version
    saturation_rate DOUBLE PRECISION
);

CREATE TABLE non_detection (
    oid VARCHAR REFERENCES object(oid), -- unique identifier for this object
    fid INTEGER NOT NULL, -- Filter ID (1=g; 2=r; 3=i)
    mjd DOUBLE PRECISION NOT NULL, -- Observation Julian date at start of exposure [days]
    diffmaglim DOUBLE PRECISION, -- 5-sigma mag limit in difference image based on PSF-fit photometry [mag]
    PRIMARY KEY (oid, fid, mjd)
);

CREATE TABLE detection (
    candid BIGINT PRIMARY KEY,  -- unique identifier for the subtraction candidate
    oid VARCHAR REFERENCES object(oid),  -- unique identifier for this object
    mjd DOUBLE PRECISION NOT NULL,  -- Observation Modified Julian date at start of exposure [days]
    fid INTEGER NOT NULL,  -- Filter ID (1=g; 2=r; 3=i)
    pid FLOAT NOT NULL,  -- Processing ID for image
    diffmaglim DOUBLE PRECISION,  -- 5-sigma mag limit in difference image based on PSF-fit photometry [mag]
    isdiffpos INTEGER NOT NULL,  -- t or 1 => candidate is from positive (sci minus ref) subtraction; f or 0 => candidate is from negative (ref minus sci) subtraction
    nid INTEGER,  -- Night ID
    ra DOUBLE PRECISION NOT NULL,  -- Right Ascension of candidate; J2000 [deg]
    dec DOUBLE PRECISION NOT NULL,  -- Declination of candidate; J2000 [deg]
    magpsf DOUBLE PRECISION NOT NULL,  -- magnitude from PSF-fit photometry [mag]
    sigmapsf DOUBLE PRECISION NOT NULL,  -- 1-sigma uncertainty in magpsf [mag]
    magap DOUBLE PRECISION,  -- Aperture mag using 8 pixel diameter aperture [mag]
    sigmagap DOUBLE PRECISION,  -- 1-sigma uncertainty in magap [mag]
    distnr DOUBLE PRECISION,  -- distance to nearest source in reference image PSF-catalog within 30 arcsec [pixels]
    rb DOUBLE PRECISION,  -- RealBogus quality score; range is 0 to 1 where closer to 1 is more reliable
    rbversion VARCHAR,  -- version of RealBogus model/classifier used to assign rb quality score
    drb DOUBLE PRECISION,  -- RealBogus quality score from Deep-Learning-based classifier; range is 0 to 1 where closer to 1 is more reliable
    drbversion VARCHAR,  -- version of Deep-Learning-based classifier model used to assign RealBogus (drb) quality score
    magapbig DOUBLE PRECISION,  -- Aperture mag using 18 pixel diameter aperture [mag]
    sigmagapbig DOUBLE PRECISION,  -- 1-sigma uncertainty in magapbig [mag]
    rfid INTEGER,  -- Processing ID for reference image to facilitate archive retrieval
    magpsf_corr DOUBLE PRECISION,  -- corrected psf magnitude magnitude
    sigmapsf_corr DOUBLE PRECISION,  -- error of the corrected psf magnitude assuming no contribution from an extended component
    sigmapsf_corr_ext DOUBLE PRECISION,  -- error of the corrected psf magnitude assuming a contribution from an extended component
    corrected BOOLEAN NOT NULL,  -- whether the corrected photometry was applied
    dubious BOOLEAN NOT NULL,  -- whether the corrected photometry should be flagged
    parent_candid BIGINT,  -- parent candid if alert coming from prv_detection (null if no parent)
    has_stamp BOOLEAN NOT NULL,  -- whether the detection has an associated image stamp triplet (when parent_candid = null)
    step_id_corr VARCHAR NOT NULL  -- correction step pipeline version
);
'''

## Relations between tables
## Indexes
### All indexes in the database
context_db_index='''
CREATE INDEX ix_object_ndet ON object USING btree (ndet);
CREATE INDEX ix_object_firstmjd ON object USING btree (firstmjd);
CREATE INDEX ix_object_g_r_max ON object USING btree (g_r_max);
CREATE INDEX ix_object_g_r_mean_corr ON object USING btree (g_r_mean_corr);
CREATE INDEX ix_object_meanra ON object USING btree (meanra);
CREATE INDEX ix_object_meandec ON object USING btree (meandec);

CREATE INDEX ix_probabilities_oid ON probability USING hash (oid);
CREATE INDEX ix_probabilities_probability ON probability USING btree (probability);
CREATE INDEX ix_probabilities_ranking ON probability USING btree (ranking);
CREATE INDEX ix_classification_rank1 ON probability USING btree (ranking) WHERE (ranking = 1);

CREATE INDEX ix_feature_oid_2 ON feature USING hash (oid);

CREATE INDEX ix_allwise_dec ON allwise USING btree (dec);
CREATE INDEX ix_allwise_ra ON allwise USING btree (ra);

CREATE INDEX ix_magstats_dmdt_first ON magstat USING btree (dmdt_first);
CREATE INDEX ix_magstats_firstmjd ON magstat USING btree (firstmjd);
CREATE INDEX ix_magstats_lastmjd ON magstat USING btree (lastmjd);
CREATE INDEX ix_magstats_magmean ON magstat USING btree (magmean);
CREATE INDEX ix_magstats_magmin ON magstat USING btree (magmin);
CREATE INDEX ix_magstats_magfirst ON magstat USING btree (magfirst);
CREATE INDEX ix_magstats_ndet ON magstat USING btree (ndet);
CREATE INDEX ix_magstats_maglast ON magstat USING btree (maglast);
CREATE INDEX ix_magstats_oid ON magstat USING hash (oid);

CREATE INDEX ix_non_detection_oid ON non_detection USING hash (oid);

CREATE INDEX ix_ndetection_oid ON detection USING hash (oid);

CREATE INDEX ix_ss_ztf_candid ON ss_ztf USING btree (candid);
CREATE INDEX ix_ss_ztf_ssnamenr ON ss_ztf USING btree (ssnamenr);

CREATE INDEX ix_reference_fid ON reference USING btree (fid);
'''

### Indexes by table
context_objectTable_index='''
CREATE INDEX ix_object_ndet ON object USING btree (ndet);
CREATE INDEX ix_object_firstmjd ON object USING btree (firstmjd);
CREATE INDEX ix_object_g_r_max ON object USING btree (g_r_max);
CREATE INDEX ix_object_g_r_mean_corr ON object USING btree (g_r_mean_corr);
CREATE INDEX ix_object_meanra ON object USING btree (meanra);
CREATE INDEX ix_object_meandec ON object USING btree (meandec);
'''
context_probTable_index='''
CREATE INDEX ix_probabilities_oid ON probability USING hash (oid);
CREATE INDEX ix_probabilities_probability ON probability USING btree (probability);
CREATE INDEX ix_probabilities_ranking ON probability USING btree (ranking);
CREATE INDEX ix_classification_rank1 ON probability USING btree (ranking) WHERE (ranking = 1);
'''
context_featureTable_index='''
CREATE INDEX ix_feature_oid_2 ON feature USING hash (oid);
'''
context_allwiseTable_index='''
CREATE INDEX ix_allwise_dec ON allwise USING btree (dec);
CREATE INDEX ix_allwise_ra ON allwise USING btree (ra);
'''
context_magstatTable_index='''
CREATE INDEX ix_magstats_dmdt_first ON magstat USING btree (dmdt_first);
CREATE INDEX ix_magstats_firstmjd ON magstat USING btree (firstmjd);
CREATE INDEX ix_magstats_lastmjd ON magstat USING btree (lastmjd);
CREATE INDEX ix_magstats_magmean ON magstat USING btree (magmean);
CREATE INDEX ix_magstats_magmin ON magstat USING btree (magmin);
CREATE INDEX ix_magstats_magfirst ON magstat USING btree (magfirst);
CREATE INDEX ix_magstats_ndet ON magstat USING btree (ndet);
CREATE INDEX ix_magstats_maglast ON magstat USING btree (maglast);
CREATE INDEX ix_magstats_oid ON magstat USING hash (oid);
'''
context_nonDetTable_index='''
CREATE INDEX ix_non_detection_oid ON non_detection USING hash (oid);
'''
context_DetTable_index='''
CREATE INDEX ix_ndetection_oid ON detection USING hash (oid);
'''
content_ss_ztfTable_index='''
CREATE INDEX ix_ss_ztf_candid ON ss_ztf USING btree (candid);
CREATE INDEX ix_ss_ztf_ssnamenr ON ss_ztf USING btree (ssnamenr);
'''
content_referenceTable_index='''
CREATE INDEX ix_reference_fid ON reference USING btree (fid);
'''


## Foreign Key ### Primary Key ### Unique Key
ref_keys='''# References Keys
probability(oid) VARCHAR REFERENCES object(oid),
feature(oid) VARCHAR REFERENCES object(oid),
feature(version) VARCHAR REFERENCES feature_version(version) NOT NULL,
magstat(oid) VARCHAR REFERENCES object(oid),
non_detection(oid) VARCHAR REFERENCES object(oid),
detection(oid) VARCHAR REFERENCES object(oid),
feature_version(step_id_feature) VARCHAR REFERENCES step(step_id),
feature_version(step_id_preprocess) VARCHAR REFERENCES step(step_id)
xmatch(oid) VARCHAR REFERENCES object(oid),
gaia_ztf(oid) VARCHAR REFERENCES object(oid),
ss_ztf(oid) VARCHAR REFERENCES object(oid),
ps1_ztf(oid) VARCHAR REFERENCES object(oid),
reference(oid) VARCHAR REFERENCES object(oid),'''



### Tables with only their column names
tab_object_columns='''TABLE object, columns=[oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change]'''
tab_probability_columns='''TABLE probability, columns=[oid, class_name, classifier_name, classifier_version, probability, ranking]'''
tab_feature_columns='''TABLE feature, columns=[oid, name, value, fid, version]'''
tab_magstat_columns='''TABLE magstat, columns=[oid, fid, stellar, corrected, ndet, ndubious, dmdt_first, dm_first, sigmadm_first, dt_first, magmean, magmedian, magmax, magmin, magsigma, maglast, magfirst, magmean_corr, magmedian_corr, magmax_corr, magmin_corr, magsigma_corr, maglast_corr, magfirst_corr, firstmjd, lastmjd, step_id_corr, saturation_rate]'''
tab_non_detection_columns='''TABLE non_detection, columns=[oid, fid, mjd, diffmaglim]'''
tab_detection_columns='''TABLE detection, Columns=[candid, oid, mjd, fid, pid, diffmaglim, isdiffpos, nid, ra, dec, magpsf, sigmapsf, magap, sigmagap, distnr, rb, rbversion, drb, drbversion, magapbig, sigmagapbig, rfid, magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext, corrected, dubious, parent_candid, has_stamp, step_id_corr]'''
tab_step_columns='''TABLE step, columns=[step_id, name, version, comments, date]'''
tab_taxonomy_columns='''TABLE taxonomy, columns=[ classifier_name, classifier_version, classes]'''
tab_feature_version_columns='''TABLE feature_version, columns=[version, step_id_feature, step_id_preprocess]'''
tab_xmatch_columns='''TABLE xmatch, columns=[oid, catid, oid_catalog, dist, class_catalog, period]'''
tab_allwise_columns='''TABLE allwise, columns=[oid_catalog, ra, dec, w1mpro, w2mpro, w3mpro, w4mpro, w1sigmpro, w2sigmpro, w3sigmpro, w4sigmpro, j_m_2mass, h_m_2mass, k_m_2mass, j_msig_2mass, h_msig_2mass, k_msig_2mass]'''
tab_dataquality_columns='''TABLE dataquality, columns=[candid, oid, fid, xpos, ypos, chipsf, sky, fwhm, classtar, mindtoedge, seeratio, aimage, bimage, aimagerat, bimagerat, nneg, nbad, sumrat, scorr, dsnrms, ssnrms, magzpsci, magzpsciunc, magzpscirms, nmatches, clrcoeff, clrcounc, zpclrcov, zpmed, clrmed, clrrms, exptime]'''
tab_gaia_ztf_columns='''TABLE gaia_ztf, columns=[oid, candid, neargaia, neargaiabright, maggaia, maggaiabright, unique1]'''
tab_ss_ztf_columns='''TABLE ss_ztf, columns=[oid, candid, ssdistnr, ssmagnr, ssnamenr]'''
tab_ps1_ztf_columns='''TABLE ps1_ztf, columns=[oid, candid, objectidps1, sgmag1, srmag1, simag1, szmag1, sgscore1, distpsnr1, objectidps2, sgmag2, srmag2, simag2, szmag2, sgscore2, distpsnr2, objectidps3, sgmag3, srmag3, simag3, szmag3, sgscore3, distpsnr3, nmtchps, unique1, unique2, unique3]'''
tab_reference_columns='''TABLE reference, columns=[oid, rfid, candid, fid, rcid, field, magnr, sigmagnr, chinr, sharpnr, ranr, decnr, mjdstartref, mjdendref, nframesref]'''
tab_pipeline_columns='''TABLE pipeline, columns=[pipeline_id, step_id_corr, step_id_feat, step_id_clf, step_id_out, step_id_stamp, date]'''
tab_information_schema_columns='''TABLE information_schema.tables, columns=[table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action]'''
tab_forced_photometry_columns='''TABLE forced_photometry, columns=[pid, oid, mjd, fid, ra, dec, mag, e_mag, mag_corr, e_mag_corr, e_mag_corr_ext, isdiffpos, corrected, dubious, parent_candid, has_stamp, field, rcid, rfid, sciinpseeing, scibckgnd, scisigpix, magzpsci, magzpsciunc, magzpscirms, clrcoeff, clrcounc, exptime, adpctdif1, adpctdif2, diffmaglim, programid, procstatus, distnr, ranr, decnr, magnr, sigmagnr, chinr, sharpnr]'''





###
# Schema for the database in a dictionary format 
###

# Dictionary of tables to describe the schema and group them by different categories and versions of the information they contain
## Important Tables
schema_base = {'object':context_objectTable, 'probability': context_probTable , 'feature': context_featureTable, 'detection': context_DetTable, 'non_detection':context_nonDetTable, 'magstat': context_magstatTable}
## All Tables
schema_all = {'object':context_objectTable, 'probability': context_probTable , 'feature': context_featureTable, 'detection': context_DetTable, 'non_detection':context_nonDetTable, 'magstat': context_magstatTable, 'step': context_stepTable, 'taxonomy': context_taxonomyTable, 'feature_version': context_featureversionTable, 'xmatch': context_xmatchTable, 'allwise': context_allwiseTable, 'dataquality': context_dataqualityTable, 'gaia_ztf': context_gaia_ztfTable, 'ss_ztf': context_ss_ztfTable, 'ps1_ztf': context_ps1_ztfTable, 'reference': context_refTable, 'pipeline': context_pipelineTable, 'forced_photometry': context_forced_photometryTable}
## All Tables w/ description
# version 1 main tables and ztfs w/ description of the columns
schema_all_cntxV1 = {'object':context_objectTable_desc, 'probability': context_probTable_desc , 'feature': context_featureTable_desc,
              'detection': context_DetTable_desc, 'non_detection': context_nonDetTable_desc, 'magstat': context_magstatTable_desc,
              'step': context_stepTable, 'taxonomy': context_taxonomyTable, 'feature_version': context_featureversionTable,
              'xmatch': context_xmatchTable, 'allwise': context_allwiseTable, 'dataquality': context_dataqualityTable,
              'gaia_ztf': context_gaia_ztfTable_desc, 'ss_ztf': context_ss_ztfTable_desc, 'ps1_ztf': context_ps1_ztfTable_desc,
              'reference': context_refTable, 'pipeline': context_pipelineTable, 'forced_photometry': context_forced_photometryTable}
# version 2 all tables w/ description of the columns
schema_all_cntxV2 = {'object':context_objectTable_desc_v2, 'probability': context_probTable_desc_v2 , 'feature': context_featureTable_desc,
              'detection': context_DetTable_desc, 'non_detection': context_nonDetTable_desc, 'magstat': context_magstatTable_desc,
              'step': context_stepTable_desc, 'taxonomy': context_taxonomyTable_desc, 'feature_version': context_featureversionTable_desc,
              'xmatch': context_xmatchTable_desc_v2, 'allwise': context_allwiseTable_desc, 'dataquality': context_dataqualityTable_desc,
              'gaia_ztf': context_gaia_ztfTable_desc, 'ss_ztf': context_ss_ztfTable_desc, 'ps1_ztf': context_ps1_ztfTable_desc,
              'reference': context_refTable_desc, 'pipeline': context_pipelineTable_desc, 'forced_photometry': context_forced_photometryTable_desc}
# Schema w/ Indexes, without description of the columns
schema_all_indx = {'object':context_objectTable + context_objectTable_index, 'probability': context_probTable + context_probTable_index , 'feature': context_featureTable + context_featureTable_index,
                'detection': context_DetTable + context_DetTable_index, 'non_detection': context_nonDetTable + context_nonDetTable_index, 'magstat': context_magstatTable + context_magstatTable_index,
                'step': context_stepTable, 'taxonomy': context_taxonomyTable, 'feature_version': context_featureversionTable,
                'xmatch': context_xmatchTable, 'allwise': context_allwiseTable + context_allwiseTable_index, 'dataquality': context_dataqualityTable,
                'gaia_ztf': context_gaia_ztfTable , 'ss_ztf': context_ss_ztfTable + content_ss_ztfTable_index, 'ps1_ztf': context_ps1_ztfTable,
                'reference': context_refTable, 'pipeline': context_pipelineTable, 'forced_photometry': context_forced_photometryTable}
# version 1 all tables w/ description of the columns w/ indexes
schema_all_cntxV1_indx = {'object':context_objectTable_desc + context_objectTable_index, 'probability': context_probTable_desc + context_probTable_index , 'feature': context_featureTable_desc + context_featureTable_index,
                'detection': context_DetTable_desc + context_DetTable_index, 'non_detection': context_nonDetTable_desc + context_nonDetTable_index, 'magstat': context_magstatTable_desc + context_magstatTable_index,
                'step': context_stepTable_desc, 'taxonomy': context_taxonomyTable_desc, 'feature_version': context_featureversionTable_desc,
                'xmatch': context_xmatchTable_desc, 'allwise': context_allwiseTable_desc + context_allwiseTable_index, 'dataquality': context_dataqualityTable_desc,
                'gaia_ztf': context_gaia_ztfTable_desc , 'ss_ztf': context_ss_ztfTable_desc + content_ss_ztfTable_index, 'ps1_ztf': context_ps1_ztfTable_desc,
                'reference': context_refTable_desc, 'pipeline': context_pipelineTable_desc, 'forced_photometry': context_forced_photometryTable_desc}
schema_all_cntxV2_indx = {'object':context_objectTable_desc_v2 + context_objectTable_index, 'probability': context_probTable_desc_v2 + context_probTable_index , 'feature': context_featureTable_desc + context_featureTable_index,
                'detection': context_DetTable_desc + context_DetTable_index, 'non_detection': context_nonDetTable_desc + context_nonDetTable_index, 'magstat': context_magstatTable_desc + context_magstatTable_index,
                'step': context_stepTable_desc, 'taxonomy': context_taxonomyTable_desc, 'feature_version': context_featureversionTable_desc,
                'xmatch': context_xmatchTable_desc_v2, 'allwise': context_allwiseTable_desc + context_allwiseTable_index, 'dataquality': context_dataqualityTable_desc,
                'gaia_ztf': context_gaia_ztfTable_desc , 'ss_ztf': context_ss_ztfTable_desc + content_ss_ztfTable_index, 'ps1_ztf': context_ps1_ztfTable_desc,
                'reference': context_refTable_desc, 'pipeline': context_pipelineTable_desc, 'forced_photometry': context_forced_photometryTable_desc}
# Schema w/ Indexes and Foreign Keys


## Schema with only the columns
schema_columns = {'object':tab_object_columns, 'probability': tab_probability_columns ,
                  'feature': tab_feature_columns, 'detection': tab_detection_columns, 'non_detection':tab_non_detection_columns,
                  'magstat': tab_magstat_columns, 'step': tab_step_columns, 'taxonomy': tab_taxonomy_columns,
                  'feature_version': tab_feature_version_columns, 'xmatch': tab_xmatch_columns, 'allwise': tab_allwise_columns,
                  'dataquality': tab_dataquality_columns, 'gaia_ztf': tab_gaia_ztf_columns, 'ss_ztf': tab_ss_ztf_columns,
                  'ps1_ztf': tab_ps1_ztf_columns, 'reference': tab_reference_columns, 'pipeline': tab_pipeline_columns, 
                  'information_schema.tables': tab_information_schema_columns, 'forced_photometry': tab_forced_photometry_columns}





# Features Description
## All features in the database in one string
context_features='''
Amplitude: Half of the difference between the median of the maximum 5% and of the minimum 5% magnitudes
AndersonDarling: Test of whether a sample of data comes from a population with a specific distribution (in this case a normal distribution)
Autocor_length: Lag value where the auto-correlation function becomes smaller than Eta_e
Beyond1Std: Percentage of points with photometric mag that lie beyond 1 sigma from the mean
Con: Number of three consecutive data points brighter/fainter than 2 sigma of the light curve
delta_mag_fid: Difference between maximum and minimum observed magnitude in a given band
delta_mjd_fid: Total timespan of the light curve in a given band
delta_period: Absolute value of the difference between the Multiband_period and the MHAOV period obtained using a single band
dmag_first_det_fid: Difference between the last non-detection diffmaglim in band "x" before the first detection in any band and the first detected magnitude in band "x"'
dmag_non_det_fid: Difference between the median non-detection diffmaglim in the "x" band before the first detection and the minimum detected magnitude (peak) in the "x" band
Eta_e: Ratio of the mean of the squares of successive mag differences to the variance of the light curve
ExcessVar: Measure of the intrinsic variability amplitude [(Std - average photometric error)/Mean]
first_mag: magnitude of the first alert in a given band
g-r_max: g-r color obtained using the brightest lc_diff (difference light curve) magnitude in each band
g-r_max_corr: g-r color obtained using the brightest lc_corr (corrected light curve or total magnitude light curve) magnitude in each band
g-r_mean: g-r color obtained using the mean lc_diff magnitude of each band
g-r_mean_corr: g-r color obtained using the mean lc_corr magnitude of each band
g-W2: color computed using the mean lc_corr g band magnitude (or the mean g band lc_diff if the source cannot be corrected) and the W2 band of AllWISE
g-W3: color computed using the mean lc_corr g band magnitude (or the mean g band lc_diff if the source cannot be corrected) and the W3 band of AllWISE
gal_b: Galactic latitude
gal_l: Galactic longitude
GP_DRW_sigma: Amplitude of the variability at short timescales (t << tau), from DRW modeling
GP_DRW_tau: Relaxation time (tau) from DRW modeling
Gskew: Median-based measure of the skew
Harmonics_mag_1: Amplitude of the 1st component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_mag_2: Amplitude of the 2nd component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_mag_3: Amplitude of the 3rd component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_mag_4: Amplitude of the 4th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_mag_5: Amplitude of the 5th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_mag_6: Amplitude of the 6th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_mag_7: Amplitude of the 7th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_mse: Mean squarre error of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_phase_2: Phase of the 2nd component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_phase_3: Phase of the 3rd component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_phase_4: Phase of the 4th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_phase_5: Phase of the 5th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_phase_6: Phase of the 6tth component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
Harmonics_phase_7: Phase of the 7th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)
IAR_phi: Level of autocorrelation using a  discrete-time representation of a DRW model
last_diffmaglim_before_fid: Last non-detection diffmaglim in the ''x'' band before the first detection in any band
last_mjd_before_fid: Last non-detection Modified Julian Date (MJD) in the ''x'' band before the first detection in any band.
LinearTrend: Slope of a linear fit to the light curve
max_diffmaglim_after_fid: Maximum non-detection diffmaglim in the ''x'' band after the first detection in any band
max_diffmaglim_before_fid: Maximum non-detection diffmaglim in the ''x'' band before the first detection in any band
MaxSlope: Maximum absolute magnitude slope between two consecutive observations
Mean: Mean lc_corr magnitude (or mean lc_diff if the source cannot be corrected)
Meanvariance: Ratio of the standard deviation to the mean magnitude
median_diffmaglim_after_fid: Median non-detection diffmaglim in the ''x'' band after the first detection in any band
median_diffmaglim_before_fid: Median non-detection diffmaglim in the ''x'' band before the first detection in any band
MedianAbsDev: Median discrepancy of the data from the median data
MedianBRP: Fraction of photometric points within amplitude/10 of the median mag
MHPS_high: Variance associated with a 10 day timescale obtained from a MHPS analysis
MHPS_low: Variance associated with a 100 day timescale obtained from a MHPS analysis
MHPS_non_zero: Number of points in the light curve used for the MHPS analysis
MHPS_PN_flag: Flag that reports whether the Poisson Noise is higher than the MHPS_high variance
MHPS_ratio: Ratio between the MHPS_low and MHPS_high variances for a given band
min_mag: minimun magnitude of the alert light curve in a given band
Multiband_period: Period obtained using the multiband MHAOV periodogram
n_det: number of detections in the alert light curve of a given band
n_neg: number of negative detections in the alert light curve (isdiffpos=-1)
n_non_det_after_fid: Number of non-detections in the ''x'' band after the first detection in any band
n_non_det_before_fid: Number of non-detections in the ''x'' band before the first detection in any band
n_pos: number of positive detections in the alert light curve (isdiffpos=+1)
PairSlopeTrend: Fraction of increasing first differences minus fraction of decreasing first differences over the last 30 time-sorted mag measures
PercentAmplitude: Largest percentage difference between either max or min mag and median mag
Period_band: Single band period computed using a Multi Harmonic Analysis of Variance (MHAOV) periodogram
positive_fraction: Fraction of detections in the difference-images of a given band which are brighter than the template image
Power_rate_1/2: Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for P/2
Power_rate_1/3: Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for P/3
Power_rate_1/4: Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for P/4
Power_rate_2: Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for 2*P
Power_rate_3: Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for 3*P
Power_rate_4: Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for 4*P
PPE: Multiband Periodogram Pseudo Entropy
Psi_CS: Range of a cumulative sum applied to the phase-folded light curve
Psi_eta: Eta_e index calculated from the folded light curve
Pvar: Probability that the source is intrinsically variable
Q31: Difference between the 3rd and the 1st quartile of the light curve
r-W2: Color computed using the mean lc_corr r band magnitude (or the mean r band lc_diff if the source cannot be corrected) and the W2 band of AllWISE
r-W3: Color computed using the mean lc_corr r band magnitude (or the mean r band lc_diff if the source cannot be corrected) and the W3 band of AllWISE
rb: Median rb (real-bogus) parameter from the ZTF alerts
Rcs: Range of a cumulative sum
SF_ML_amplitude: Rms magnitude difference of the structure function, computed over a 1 yr timescale
SF_ML_gamma: Logarithmic gradient of the mean change in magnitude (computed from the structure function)
sgscore1: Morphological star/galaxy score of the closest source from PanSTARRS1 (values closer to 1 imply a higher likelihood of the source being a star)
Skew: Skewness measure
SmallKurtosis: Small sample kurtosis of the magnitudes
SPM_A: Supernova parametric model  A
SPM_beta: Supernova parametric model beta
SPM_chi: Supernova parametric model reduced chi2 of the light curve fit
SPM_gamma: Supernova parametric model gamma
SPM_t0: Supernova parametric model t0
SPM_tau_fall: Supernova parametric model fall time
SPM_tau_rise: Supernova parametric model rise time
Std: Standard deviation of the light curve
StetsonK: Robust kurtosis measure
W1-W2: color computed using the W1 and W2 bands of AllWISE
W2-W3: color computed using the W2 and W3 bands of AllWISE
'''

## features description
Amplitude= '''Half of the difference between the median of the maximum 5% and of the minimum 5% magnitudes'''
AndersonDarling= '''Test of whether a sample of data comes from a population with a specific distribution (in this case a normal distribution)'''
Autocor_length= '''Lag value where the auto-correlation function becomes smaller than Eta_e'''
Beyond1Std= '''Percentage of points with photometric mag that lie beyond 1 sigma from the mean'''
Con= '''Number of three consecutive data points brighter/fainter than 2 sigma of the light curve'''
delta_mag_fid= '''Difference between maximum and minimum observed magnitude in a given band'''
delta_mjd_fid= '''Total timespan of the light curve in a given band'''
delta_period= '''Absolute value of the difference between the Multiband_period and the MHAOV period obtained using a single band'''
dmag_first_det_fid= '''Difference between the last non-detection diffmaglim in band "x" before the first detection in any band and the first detected magnitude in band "x"' '''
dmag_non_det_fid= '''Difference between the median non-detection diffmaglim in the "x" band before the first detection and the minimum detected magnitude (peak) in the "x" band'''
Eta_e= '''Ratio of the mean of the squares of successive mag differences to the variance of the light curve'''
ExcessVar= '''Measure of the intrinsic variability amplitude [(Std - average photometric error)/Mean]'''
first_mag= '''magnitude of the first alert in a given band'''
g_r_max= '''g-r color obtained using the brightest lc_diff (difference light curve) magnitude in each band'''
g_r_max_corr= '''g-r color obtained using the brightest lc_corr (corrected light curve or total magnitude light curve) magnitude in each band'''
g_r_mean= '''g-r color obtained using the mean lc_diff magnitude of each band'''
g_r_mean_corr= '''g-r color obtained using the mean lc_corr magnitude of each band'''
g_W2= '''color computed using the mean lc_corr g band magnitude (or the mean g band lc_diff if the source cannot be corrected) and the W2 band of AllWISE '''
g_W3= '''color computed using the mean lc_corr g band magnitude (or the mean g band lc_diff if the source cannot be corrected) and the W3 band of AllWISE '''
gal_b= '''Galactic latitude '''
gal_l= '''Galactic longitude'''
GP_DRW_sigma= '''Amplitude of the variability at short timescales (t << tau), from DRW modeling'''
GP_DRW_tau= '''Relaxation time (tau) from DRW modeling '''
Gskew= '''Median-based measure of the skew'''
Harmonics_mag_1= '''Amplitude of the 1st component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_mag_2= '''Amplitude of the 2nd component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_mag_3= '''Amplitude of the 3rd component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_mag_4= '''Amplitude of the 4th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_mag_5= '''Amplitude of the 5th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_mag_6= '''Amplitude of the 6th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_mag_7= '''Amplitude of the 7th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_mse= '''Mean squarre error of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_phase_2= '''Phase of the 2nd component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_phase_3= '''Phase of the 3rd component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_phase_4= '''Phase of the 4th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_phase_5= '''Phase of the 5th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_phase_6= '''Phase of the 6tth component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
Harmonics_phase_7= '''Phase of the 7th component of the harmonics series (obtained by fitting a harmonic series up to the seventh harmonic)'''
IAR_phi= '''Level of autocorrelation using a  discrete-time representation of a DRW model'''
last_diffmaglim_before_fid= '''Last non-detection diffmaglim in the ''x'' band before the first detection in any band'''
last_mjd_before_fid= '''Last non-detection Modified Julian Date (MJD) in the ''x'' band before the first detection in any band.'''
LinearTrend= '''Slope of a linear fit to the light curve'''
max_diffmaglim_after_fid= '''Maximum non-detection diffmaglim in the ''x'' band after the first detection in any band'''
max_diffmaglim_before_fid= '''Maximum non-detection diffmaglim in the ''x'' band before the first detection in any band'''
MaxSlope= '''Maximum absolute magnitude slope between two consecutive observations'''
Mean= '''Mean lc_corr magnitude (or mean lc_diff if the source cannot be corrected)'''
Meanvariance= '''Ratio of the standard deviation to the mean magnitude'''
median_diffmaglim_after_fid= '''Median non-detection diffmaglim in the ''x'' band after the first detection in any band'''
median_diffmaglim_before_fid= '''Median non-detection diffmaglim in the ''x'' band before the first detection in any band'''
MedianAbsDev= '''Median discrepancy of the data from the median data'''
MedianBRP= '''Fraction of photometric points within amplitude/10 of the median mag'''
MHPS_high= '''Variance associated with a 10 day timescale obtained from a MHPS analysis'''
MHPS_low= '''Variance associated with a 100 day timescale obtained from a MHPS analysis'''
MHPS_non_zero= '''Number of points in the light curve used for the MHPS analysis'''
MHPS_PN_flag= '''Flag that reports whether the Poisson Noise is higher than the MHPS_high variance'''
MHPS_ratio= '''Ratio between the MHPS_low and MHPS_high variances for a given band'''
min_mag= '''minimun magnitude of the alert light curve in a given band'''
Multiband_period= '''Period obtained using the multiband MHAOV periodogram'''
n_det= '''number of detections in the alert light curve of a given band'''
n_neg= '''number of negative detections in the alert light curve (isdiffpos=-1)'''
n_non_det_after_fid= '''Number of non-detections in the ''x'' band after the first detection in any band'''
n_non_det_before_fid= '''Number of non-detections in the ''x'' band before the first detection in any band'''
n_pos= '''number of positive detections in the alert light curve (isdiffpos=+1)'''
PairSlopeTrend= '''Fraction of increasing first differences minus fraction of decreasing first differences over the last 30 time-sorted mag measures'''
PercentAmplitude= '''Largest percentage difference between either max or min mag and median mag'''
Period_band= '''Single band period computed using a Multi Harmonic Analysis of Variance (MHAOV) periodogram'''
positive_fraction= '''Fraction of detections in the difference-images of a given band which are brighter than the template image'''
Power_rate_1_2= '''Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for P/2'''
Power_rate_1_3= '''Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for P/3'''
Power_rate_1_4= '''Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for P/4'''
Power_rate_2= '''Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for 2*P'''
Power_rate_3= '''Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for 3*P'''
Power_rate_4= '''Ratio between the power of the multiband periodogram obtained for the best period candidate (P) and for 4*P'''
PPE= '''Multiband Periodogram Pseudo Entropy'''
Psi_CS= '''Range of a cumulative sum applied to the phase-folded light curve'''
Psi_eta= '''Eta_e index calculated from the folded light curve'''
Pvar= '''Probability that the source is intrinsically variable'''
Q31= '''Difference between the 3rd and the 1st quartile of the light curve'''
r_W2= '''Color computed using the mean lc_corr r band magnitude (or the mean r band lc_diff if the source cannot be corrected) and the W2 band of AllWISE '''
r_W3= '''Color computed using the mean lc_corr r band magnitude (or the mean r band lc_diff if the source cannot be corrected) and the W3 band of AllWISE '''
rb= '''Median rb (real-bogus) parameter from the ZTF alerts'''
Rcs= '''Range of a cumulative sum'''
SF_ML_amplitude= '''Rms magnitude difference of the structure function, computed over a 1 yr timescale'''
SF_ML_gamma= '''Logarithmic gradient of the mean change in magnitude (computed from the structure function)'''
sgscore1= '''Morphological star/galaxy score of the closest source from PanSTARRS1 (values closer to 1 imply a higher likelihood of the source being a star)'''
Skew= '''Skewness measure'''
SmallKurtosis= '''Small sample kurtosis of the magnitudes'''
SPM_A= '''Supernova parametric model  A'''
SPM_beta= '''Supernova parametric model beta'''
SPM_chi= '''Supernova parametric model reduced chi2 of the light curve fit'''
SPM_gamma= '''Supernova parametric model gamma'''
SPM_t0= '''Supernova parametric model t0'''
SPM_tau_fall= '''Supernova parametric model fall time'''
SPM_tau_rise= '''Supernova parametric model rise time'''
Std= '''Standard deviation of the light curve'''
StetsonK= '''Robust kurtosis measure'''
W1_W2= '''color computed using the W1 and W2 bands of AllWISE '''
W2_W3= '''color computed using the W2 and W3 bands of AllWISE '''




# database tables description
alerce_tables_desc = '''
TABLE "object": contains basic filter and time-aggregated statistics such as location, number of observations, and the times of first and last detection.
TABLE "probability": classification probabilities associated to a given object, classifier, and class. Contain the object classification probabilities, including those from the stamp and light curve classifiers, and from different versions of these classifiers.
TABLE "feature": contains the object light curve statistics and other features used for ML classification and which are stored as json files in our database.
TABLE "magstat": contains time-aggregated statistics separated by filter, such as the average magnitude, the initial magnitude change rate, number of detections, etc.
TABLE "non_detection": contains the limiting magnitudes of previous non-detections separated by filter.
TABLE "detection": contains the object light curves including their difference and corrected magnitudes and associated errors separated by filter.
TABLE "step": contains the different pipeline steps and their versions.
TABLE "taxonomy": contains details about the different taxonomies used in our stamp and light curve classifiers, which can evolve with time.
TABLE "feature_version": contains the version of the feature extraction and preprocessing steps used to generate the features.
TABLE "xmatch": contains the object cross-matches and associated cross-match catalogs.
TABLE "allwise": contains the AllWISE catalog information for the objects.
TABLE "dataquality": detailed object information regarding the quality of the data
TABLE "gaia_ztf": GAIA objects near detected ZTF objects
TABLE "ss_ztf": known solar system objects near detected objects
TABLE "ps1_ztf": PanSTARRS objects near detected ZTF objects
TABLE "reference": properties of the reference images used to build templates
TABLE "pipeline": information about the different pipeline steps and their versions
TABLE "information_schema.tables": information about the database tables and columns
TABLE "forced_photometry": contains the forced photometry measurements for each object, including the object position, magnitude, and associated errors, and the photometry of the reference image.
'''

# tables description without object and probability tables
## version 2
alerce_tables_desc_v2 = '''
TABLE "feature": contains the object light curve statistics and other features used for ML classification and which are stored as json files in our database.
TABLE "magstat": contains time-aggregated statistics separated by filter, such as the average magnitude, the initial magnitude change rate, number of detections, etc.
TABLE "non_detection": contains the limiting magnitudes of previous non-detections separated by filter.
TABLE "detection": contains the object light curves including their difference and corrected magnitudes and associated errors separated by filter.
TABLE "step": contains the different pipeline steps and their versions.
TABLE "taxonomy": contains details about the different taxonomies used in our stamp and light curve classifiers, which can evolve with time.
TABLE "feature_version": contains the version of the feature extraction and preprocessing steps used to generate the features.
TABLE "xmatch": contains the object cross-matches and associated cross-match catalogs.
TABLE "allwise": contains the AllWISE catalog information for the objects.
TABLE "dataquality": detailed object information regarding the quality of the data
TABLE "gaia_ztf": GAIA objects near detected ZTF objects
TABLE "ss_ztf": known solar system objects near detected objects
TABLE "ps1_ztf": PanSTARRS objects near detected ZTF objects
TABLE "reference": properties of the reference images used to build templates
TABLE "pipeline": information about the different pipeline steps and their versions
TABLE "information_schema.tables": information about the database tables and columns
TABLE "forced_photometry": contains the forced photometry measurements for each object, including the object position, magnitude, and associated errors, and the photometry of the reference image.
'''
# tables description without object and probability tables
## version 3
alerce_tables_desc_v3 = '''
TABLE "feature": contains the object light curve statistics and other features used for ML classification and which are stored as json files in our database.
TABLE "magstat": contains time-aggregated statistics separated by filter, such as the average magnitude, the initial magnitude change rate, number of detections, etc.
TABLE "non_detection": contains the limiting magnitudes of previous non-detections separated by filter.
TABLE "detection": contains the object light curves including their difference and corrected magnitudes and associated errors separated by filter.
TABLE "step": steps Pipeline versions and information
TABLE "taxonomy": contains details about the different taxonomies used in our stamp and light curve classifiers, which can evolve with time.
TABLE "feature_version": feature processing version with its feature and preprocess step docker id 
TABLE "xmatch": contains the object cross-matches and associated cross-match catalogs.
TABLE "allwise": AllWISE Source Catalog
TABLE "dataquality": detailed object information regarding the quality of the data
TABLE "gaia_ztf": GAIA objects near detected ZTF objects
TABLE "ss_ztf": known solar system objects near detected objects
TABLE "ps1_ztf": PanSTARRS objects near detected ZTF objects
TABLE "reference": properties of the reference images used to build templates
TABLE "pipeline": Informatin about ALeRCE's pipeline version, steps and dates
TABLE "information_schema.tables": information about the database tables and columns
TABLE "forced_photometry": contains the forced photometry measurements for each object, including the object position, magnitude, and associated errors, and the photometry of the reference image.
'''