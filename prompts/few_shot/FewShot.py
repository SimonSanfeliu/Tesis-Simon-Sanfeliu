# Description: This file contains the prompts and queries for the few-shot and few-shot w/ CoT examples of the ALeRCE DB.

# Few-Shot examples

### Simple query example
#### Prompt from usecase with index: 26, difficulty simple, type object   # tokens:268
simple_ex1='''# Postgres SQL tables, with their properties:

CREATE TABLE probability ( /* this table contains the machine learning derived classification probabilities and rankings */
    oid VARCHAR REFERENCES object(oid),
    class_name VARCHAR, /* name of the class */
    classifier_name VARCHAR, /* name of the classifier */
    classifier_version VARCHAR, /* version of the classiifer */
    probability DOUBLE PRECISION NOT NULL, /* probability of the class given a classifier and version */
    ranking INTEGER NOT NULL, /* class probability ranking (1 is the most likely class) */
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);
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
    meanra DOUBLE PRECISION,  /* mean right ascencion */
    meandec DOUBLE PRECISION,  /* mean declination */
    sigmara DOUBLE PRECISION, /* right ascension dispersion */
    sigmadec DOUBLE PRECISION, /* declination dispersion */
    deltajd DOUBLE PRECISION, /* time difference between last and first detection */
    firstmjd DOUBLE PRECISION, /* time of first detection */
    lastmjd DOUBLE PRECISION, /* time of last detection */
    step_id_corr VARCHAR,
    diffpos BOOLEAN, /* whether the first detection was positive or negative */
    reference_change BOOLEAN /* whether the reference image changes */
);

    # Important Information for the query

-- mjd date for August 25 2022 = 59816.0
-- mjd date for September 10 2022 = 59832.0
-- object.firstmjd represents the time of the first detection in the given fid in days
-- object.ndet represents the total number of detections for the object


    # Request: Query objects which appeared between august 25 and september 10 of 2022, which are classified as supernova by the stamp classifier with a probability greater than 0.5 or which have more than one detection
    '''

simple_ex1_query ='''SELECT
      *
  FROM
      object
  INNER JOIN
      probability
  ON
      object.oid = probability.oid
  WHERE
      object.firstMJD > 59816.0
      AND object.firstMJD < 59832.0
      AND probability.class_name='SN'
      AND probability.classifier_name='stamp_classifier'
      AND probability.ranking=1
      AND (probability.probability>0.5
          OR object.ndet>1)
 '''

#### Prompt from usecase with index: 23, difficulty simple, type object   # tokens:217
simple_ex2='''# Postgres SQL tables, with their properties:

CREATE TABLE probability ( /* this table contains the machine learning derived classification probabilities and rankings */
    oid VARCHAR REFERENCES object(oid),
    class_name VARCHAR, /* name of the class */
    classifier_name VARCHAR, /* name of the classifier */
    classifier_version VARCHAR, /* version of the classiifer */
    probability DOUBLE PRECISION NOT NULL, /* probability of the class given a classifier and version */
    ranking INTEGER NOT NULL, /* class probability ranking (1 is the most likely class) */
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);
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
    meanra DOUBLE PRECISION,  /* mean right ascencion */
    meandec DOUBLE PRECISION,  /* mean declination */
    sigmara DOUBLE PRECISION, /* right ascension dispersion */
    sigmadec DOUBLE PRECISION, /* declination dispersion */
    deltajd DOUBLE PRECISION, /* time difference between last and first detection */
    firstmjd DOUBLE PRECISION, /* time of first detection */
    lastmjd DOUBLE PRECISION, /* time of last detection */
    step_id_corr VARCHAR,
    diffpos BOOLEAN, /* whether the first detection was positive or negative */
    reference_change BOOLEAN /* whether the reference image changes */
);

    # Important Information for the query

-- it is important to return all the probabilites (different rankings)


    # Request: I want to find the probabilities of the objects most likely to be either "SNIa", "SNIbc", "SNII", "SLSN" classified by the lc_classifier with a probability higher than 0.2
    '''
simple_ex2_query ='''SELECT
        *
    FROM
        object INNER JOIN probability
        ON object.oid=probability.oid
    WHERE
        probability.classifier_name='lc_classifier'
        AND object.oid IN
    (
    SELECT
        oid
    FROM
        probability
    WHERE
        classifier_name='lc_classifier'
        AND class_name IN ('SNIa', 'SNIbc', 'SNII', 'SLSN')
        AND ranking=1
        AND probability > 0.2
    )
'''

#### Prompt from usecase with index: 33, difficulty simple, type other    # tokens:350
simple_ex3='''# Postgres SQL tables, with their properties:

CREATE TABLE detection (  /* this table contains information about the object detections, or its light curve. Avoid doing filters on this table in any parameter other than oid */
    candid BIGINT PRIMARY KEY, /* unique candidate identifier */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    mjd DOUBLE PRECISION NOT NULL, /* time of detection in modified julian date */
    fid INTEGER NOT NULL, /* filter identifier */
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
);

    # Important Information for the query

-- oid represents the unique identifier for this object
-- candid represents the	unique identifier for the subtraction candidate
-- mjd represents the	Observation Modified Julian date at start of exposure [days]
-- fid represents the Filter ID (1=g; 2=r; 3=i)
-- magpsf	represents the magnitude from PSF-fit photometry [mag]


    # Request: Given the next list of oids ['ZTF21aaqosfj' 'ZTF21aaqoski' 'ZTF21aaqoskz' 'ZTF21aaqosly' 'ZTF21aaqosmj' 'ZTF21aaqosmi' 'ZTF21aaqosmk' 'ZTF21aaqosmm' 'ZTF21aaqosmp' 'ZTF21aaqosmq'], give a query that returns the magnitudes of their detections, with their oids, candidate identifiers, the date of exposure and the filter id.
    '''
simple_ex3_query='''SELECT oid, candid, mjd, fid, magpsf
    FROM detection
    WHERE
    oid in ('ZTF21aaqosfj','ZTF21aaqoski','ZTF21aaqoskz','ZTF21aaqosly','ZTF21aaqosmj','ZTF21aaqosmi','ZTF21aaqosmk','ZTF21aaqosmm','ZTF21aaqosmp','ZTF21aaqosmq')
'''

### Medium query example
#### Prompt from usecase with index: 10, difficulty medium, type object   # tokens:211
medium_ex='''
 # Postgres SQL tables, with their properties:

CREATE TABLE probability ( /* this table contains the machine learning derived classification probabilities and rankings */
    oid VARCHAR REFERENCES object(oid),
    class_name VARCHAR, /* name of the class */
    classifier_name VARCHAR, /* name of the classifier */
    classifier_version VARCHAR, /* version of the classiifer */
    probability DOUBLE PRECISION NOT NULL, /* probability of the class given a classifier and version */
    ranking INTEGER NOT NULL, /* class probability ranking (1 is the most likely class) */
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);
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
    meanra DOUBLE PRECISION,  /* mean right ascencion */
    meandec DOUBLE PRECISION,  /* mean declination */
    sigmara DOUBLE PRECISION, /* right ascension dispersion */
    sigmadec DOUBLE PRECISION, /* declination dispersion */
    deltajd DOUBLE PRECISION, /* time difference between last and first detection */
    firstmjd DOUBLE PRECISION, /* time of first detection */
    lastmjd DOUBLE PRECISION, /* time of last detection */
    step_id_corr VARCHAR,
    diffpos BOOLEAN, /* whether the first detection was positive or negative */
    reference_change BOOLEAN /* whether the reference image changes */
);

    # Request: Get the object identifiers and probabilities in the light curve classifier for objects classified in the light curve classifier as SNIa with ranking=1 and CV/Nova with ranking=2, where the difference between the probabilities at each ranking is lower than 0.1. Return oids, and the probability for each class
    '''
medium_ex_query='''
SELECT
  sq1.oid, sq1.probability as SN_prob, sq2.probability as CV_prob
FROM
  (
SELECT
    object.oid, probability.probability
FROM
    object INNER JOIN
    probability
    ON object.oid = probability.oid
WHERE
    probability.classifier_name='lc_classifier'
    AND probability.class_name='SNIa'
    AND probability.ranking = 1
) AS sq1
  INNER JOIN (
SELECT
    object.oid, probability.probability
FROM
    object INNER JOIN
    probability
    ON object.oid = probability.oid
WHERE
    probability.classifier_name='lc_classifier'
    AND probability.class_name='CV/Nova'
    AND probability.ranking = 2
) as sq2
  ON sq1.oid = sq2.oid
WHERE
  sq1.probability - sq2.probability < 0.1
'''

### Advanced query example
#### Prompt from usecase with index: 15, difficulty advanced, type other
adv_ex='''# Postgres SQL tables, with their properties:

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
    meanra DOUBLE PRECISION,  /* mean right ascencion */
    meandec DOUBLE PRECISION,  /* mean declination */
    sigmara DOUBLE PRECISION, /* right ascension dispersion */
    sigmadec DOUBLE PRECISION, /* declination dispersion */
    deltajd DOUBLE PRECISION, /* time difference between last and first detection */
    firstmjd DOUBLE PRECISION, /* time of first detection */
    lastmjd DOUBLE PRECISION, /* time of last detection */
    step_id_corr VARCHAR,
    diffpos BOOLEAN, /* whether the first detection was positive or negative */
    reference_change BOOLEAN /* whether the reference image changes */
);
CREATE TABLE probability ( /* this table contains the machine learning derived classification probabilities and rankings */
    oid VARCHAR REFERENCES object(oid),
    class_name VARCHAR, /* name of the class */
    classifier_name VARCHAR, /* name of the classifier */
    classifier_version VARCHAR, /* version of the classiifer */
    probability DOUBLE PRECISION NOT NULL, /* probability of the class given a classifier and version */
    ranking INTEGER NOT NULL, /* class probability ranking (1 is the most likely class) */
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);
CREATE TABLE detection (  /* this table contains information about the object detections, or its light curve. Avoid doing filters on this table in any parameter other than oid */
    candid BIGINT PRIMARY KEY, /* unique candidate identifier */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    mjd DOUBLE PRECISION NOT NULL, /* time of detection in modified julian date */
    fid INTEGER NOT NULL, /* filter identifier */
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
);
CREATE TABLE magstat ( /* different statistics for the object divided by band or filter */
    oid VARCHAR REFERENCES object(oid), /* unique object identifier */
    fid INTEGER NOT NULL, /* band or filter identifier */
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
);

    # Important Information for the query

-- mjd date for September 01 = 60188.0
-- mjd date for September 02 = 60189.0
-- firstmjd represents the time of the first detection
-- dmdt_first represents the initial rise estimate, the maximum slope between the first detection and any previous non-detection
-- dmdt_first values are negatives


-- A fast riser is defined as an object whose first detection is brighter by at least a rate of 0.25 magnitudes/day than the last non-detection obtained before the first detection (both first detection and last non-detection are in the same filter)


    # Request: Get the object identifiers, probabilities in the stamp classifier and light curves (only detections) for objects whose highest probability in the stamp classifier is obtained for class SN, that had their first detection in the first 2 days of september, and that qualify as fast risers.
    '''
adv_ex_query='''SELECT
        sq.oid, sq.probability, sq.candid, sq.fid, sq.mjd,
        magstat.fid as magstat_fid, magstat.dmdt_first
    FROM
    (
    SELECT
    detection.oid, detection.candid, detection.fid, detection.mjd,
    obj_oids.probability
    FROM
    (
    SELECT
        object.oid, probability.probability
    FROM
        object INNER JOIN
        probability
        ON object.oid = probability.oid
    WHERE
        probability.classifier_name='stamp_classifier'
        AND probability.class_name='SN'
        AND probability.ranking=1
        AND object.firstmjd > 60188.0
        AND object.firstmjd < 60189.0
    ) as obj_oids
        INNER JOIN
        detection ON detection.oid = obj_oids.oid
    ) AS sq
    INNER JOIN magstat
    ON sq.oid = magstat.oid
    WHERE
    magstat.dmdt_first < -0.25
    ORDER BY oid
'''



# Few-Shot example w/ CoT

### Simple query example
#### Prompt from usecase with index: 26, difficulty simple, type object   # tokens:268 
simple_ex1_cot='''# Postgres SQL tables, with their properties:

CREATE TABLE probability ( /* this table contains the machine learning derived classification probabilities and rankings */
    oid VARCHAR REFERENCES object(oid),
    class_name VARCHAR, /* name of the class */
    classifier_name VARCHAR, /* name of the classifier */
    classifier_version VARCHAR, /* version of the classiifer */
    probability DOUBLE PRECISION NOT NULL, /* probability of the class given a classifier and version */
    ranking INTEGER NOT NULL, /* class probability ranking (1 is the most likely class) */
    PRIMARY KEY (oid, class_name, classifier_name, classifier_version)
);
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
    meanra DOUBLE PRECISION,  /* mean right ascencion */
    meandec DOUBLE PRECISION,  /* mean declination */
    sigmara DOUBLE PRECISION, /* right ascension dispersion */
    sigmadec DOUBLE PRECISION, /* declination dispersion */
    deltajd DOUBLE PRECISION, /* time difference between last and first detection */
    firstmjd DOUBLE PRECISION, /* time of first detection */
    lastmjd DOUBLE PRECISION, /* time of last detection */
    step_id_corr VARCHAR,
    diffpos BOOLEAN, /* whether the first detection was positive or negative */
    reference_change BOOLEAN /* whether the reference image changes */
);

    # Important Information for the query

-- mjd date for August 25 2022 = 59816.0
-- mjd date for September 10 2022 = 59832.0
-- object.firstmjd represents the time of the first detection in the given fid in days
-- object.ndet represents the total number of detections for the object


    # Request: Query objects which appeared between august 25 and september 10 of 2022, which are classified as supernova by the stamp classifier with a probability greater than 0.5 or which have more than one detection

    REQUEST: Query objects which appeared between august 25 and september 10 of 2022, which are classified as supernova by the stamp classifier with a probability greater than 0.5 or which have more than one detection
    Let's Think Step By Step. The user is asking for general information about objects between two specific dates, information that is in the object table. Also, the user is asking for objects classified by supernova with a specific probability, so we need to use the probability table. 
    Finally, the user is asking for objects with more than one detection, so we can use the object table again to add this condition with an OR statement from the requested probability.
    '''



#### Prompt from usecase with index: 23, difficulty simple, type object   # tokens:217
simple_ex2_cot='''
 # Postgres SQL tables, with their properties:
    
TABLE probability, columns=[oid, class_name, classifier_name, classifier_version, probability, ranking]
TABLE object, columns=[oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change]
    # Context:
    ## General information of the schema and the database
    [general_context]
    ## Information useful for the query 
    -- it is important to return all the probabilites (different rankings)
    REQUEST: I want to find the probabilities of the objects most likely to be either "SNIa", "SNIbc", "SNII", "SLSN" classified by the lc_classifier with a probability higher than 0.2
    Let's Think Step By Step. The user is
    ANSWER:
     
SELECT
    *
FROM
    object INNER JOIN probability
    ON object.oid=probability.oid
WHERE
    probability.classifier_name='lc_classifier'
    AND object.oid IN
(
SELECT
    oid
FROM
    probability
WHERE
    classifier_name='lc_classifier'
    AND class_name IN ('SNIa', 'SNIbc', 'SNII', 'SLSN')
    AND ranking=1
    AND probability > 0.2
)
'''
#### Prompt from usecase with index: 33, difficulty simple, type other    # tokens:350
simple_ex3_cot='''
 # Postgres SQL tables, with their properties:
TABLE detection, Columns=[candid, oid, mjd, fid, pid, diffmaglim, isdiffpos, nid, ra, dec, magpsf, sigmapsf, magap, sigmagap, distnr, rb, rbversion, drb, drbversion, magapbig, sigmagapbig, rfid, magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext, corrected, dubious, parent_candid, has_stamp, step_id_corr]
    # Context:
    ## General information of the schema and the database
    [general_context]
    ## Information useful for the query 
-- oid represents the unique identifier for this object
-- candid represents the unique identifier for the subtraction candidate
-- mjd represents the Observation Modified Julian date at start of exposure [days]
-- fid represents the Filter ID (1=g; 2=r; 3=i)
-- magpsf represents the magnitude from PSF-fit photometry [mag]
    REQUEST: Given the next list of oids ['ZTF21aaqosfj' 'ZTF21aaqoski' 'ZTF21aaqoskz' 'ZTF21aaqosly' 'ZTF21aaqosmj' 'ZTF21aaqosmi' 'ZTF21aaqosmk' 'ZTF21aaqosmm' 'ZTF21aaqosmp' 'ZTF21aaqosmq'], give a query that returns the magnitudes of their detections, with their oids, candidate identifiers, the date of exposure and the filter id.
    ANSWER:
     
SELECT oid, candid, mjd, fid, magpsf
FROM detection
WHERE
oid in ('ZTF21aaqosfj','ZTF21aaqoski','ZTF21aaqoskz','ZTF21aaqosly','ZTF21aaqosmj','ZTF21aaqosmi','ZTF21aaqosmk','ZTF21aaqosmm','ZTF21aaqosmp','ZTF21aaqosmq')
'''

### Medium query example
medium_ex_cot='''
 # Postgres SQL tables, with their properties:
TABLE probability, columns=[oid, class_name, classifier_name, classifier_version, probability, ranking]
TABLE object, columns=[oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change]
    # Context:
    ## General information of the schema and the database
    [general_context]
    REQUEST: Get the object identifiers and probabilities in the light curve classifier for objects classified in the light curve classifier as SNIa with ranking=1 and CV/Nova with ranking=2, where the difference between the probabilities at each ranking is lower than 0.1. Return oids, and the probability for each class
    ANSWER: SELECT
  sq1.oid, sq1.probability as SN_prob, sq2.probability as CV_prob
FROM
  (
SELECT
    object.oid, probability.probability
FROM
    object INNER JOIN
    probability
    ON object.oid = probability.oid
WHERE
    probability.classifier_name='lc_classifier'
    AND probability.class_name='SNIa'
    AND probability.ranking = 1
) AS sq1
  INNER JOIN (
SELECT
    object.oid, probability.probability
FROM
    object INNER JOIN
    probability
    ON object.oid = probability.oid
WHERE
    probability.classifier_name='lc_classifier'
    AND probability.class_name='CV/Nova'
    AND probability.ranking = 2
) as sq2
  ON sq1.oid = sq2.oid
WHERE
  sq1.probability - sq2.probability < 0.1
'''

### Advanced query example
#### Prompt from usecase with index: 15, difficulty advanced, type other 
adv_ex_cot='''
# Postgres SQL tables, with their properties:    
TABLE probability, columns=[oid, class_name, classifier_name, classifier_version, probability, ranking]
TABLE object, columns=[oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change]
TABLE detection, Columns=[candid, oid, mjd, fid, pid, diffmaglim, isdiffpos, nid, ra, dec, magpsf, sigmapsf, magap, sigmagap, distnr, rb, rbversion, drb, drbversion, magapbig, sigmagapbig, rfid, magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext, corrected, dubious, parent_candid, has_stamp, step_id_corr]
    # Context:
    ## General information of the schema and the database
    [general_context]
    ## Information useful for the query 
-- mjd date for September 01 = 60188.0
-- mjd date for September 02 = 60189.0
-- firstmjd represents the time of the first detection
-- dmdt_first represents the initial rise estimate, the maximum slope between the first detection and any previous non-detection
-- dmdt_first values are negatives
 
-- A fast riser is defined as an object whose first detection is brighter by at least a rate of 0.25 magnitudes/day than the last non-detection obtained before the first detection (both first detection and last non-detection are in the same filter)
    REQUEST: Get the object identifiers, probabilities in the stamp classifier and light curves (only detections) for objects whose highest probability in the stamp classifier is obtained for class SN, that had their first detection in the first 2 days of september, and that qualify as fast risers.
    ANSWER: SELECT
    sq.oid, sq.probability, sq.candid, sq.fid, sq.mjd,
    magstat.fid as magstat_fid, magstat.dmdt_first
FROM
  (
SELECT
  detection.oid, detection.candid, detection.fid, detection.mjd,
  obj_oids.probability
FROM
  (
SELECT
    object.oid, probability.probability
FROM
    object INNER JOIN
    probability
    ON object.oid = probability.oid
WHERE
    probability.classifier_name='stamp_classifier'
    AND probability.class_name='SN'
    AND probability.ranking=1
    AND object.firstmjd > 60188.0
    AND object.firstmjd < 60189.0
) as obj_oids
    INNER JOIN
    detection ON detection.oid = obj_oids.oid
) AS sq
  INNER JOIN magstat
  ON sq.oid = magstat.oid
WHERE
  magstat.dmdt_first < -0.25
ORDER BY oid
'''
    #   tokens:443 




 
 