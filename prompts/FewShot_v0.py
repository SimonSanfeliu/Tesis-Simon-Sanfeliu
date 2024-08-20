# Description: Few-shot examples for the FewShot in-context learning usecase

## possible structure
# [{request:, prompt:, query:, usecase:, difficulty:, type:}]

# Few-Shot examples 

### Simple query example
#### Prompt from usecase with index: 26, difficulty simple, type object   # tokens:268 
simple_ex1='''
 # Postgres SQL tables, with their properties:
    
TABLE probability, columns=[oid, class_name, classifier_name, classifier_version, probability, ranking]
TABLE object, columns=[oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change]
    # Context:
    ## General information of the schema and the database
    [general_context]
    ## Information useful for the query 
-- mjd date for August 25 2022 = 59816.0
-- mjd date for September 10 2022 = 59832.0
-- object.firstmjd represents the time of the first detection in the given fid in days
-- object.ndet represents the total number of detections for the object
    REQUEST: Query objects which appeared between august 25 and september 10 of 2022, which are classified as supernova by the stamp classifier with a probability greater than 0.5 or which have more than one detection
    ANSWER:
     
SELECT
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
simple_ex2='''
 # Postgres SQL tables, with their properties:    
    TABLE probability, columns=[oid, class_name, classifier_name, classifier_version, probability, ranking]
    TABLE object, columns=[oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change]
    # Context:
    ## General information of the schema and the database
    [general_context]
    ## Information useful for the query 
    -- it is important to return all the probabilites (different rankings)
    REQUEST: I want to find the probabilities of the objects most likely to be either "SNIa", "SNIbc", "SNII", "SLSN" classified by the lc_classifier with a probability higher than 0.2
    ANSWER: SELECT
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
simple_ex3='''
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
    ANSWER: SELECT oid, candid, mjd, fid, magpsf
    FROM detection
    WHERE
    oid in ('ZTF21aaqosfj','ZTF21aaqoski','ZTF21aaqoskz','ZTF21aaqosly','ZTF21aaqosmj','ZTF21aaqosmi','ZTF21aaqosmk','ZTF21aaqosmm','ZTF21aaqosmp','ZTF21aaqosmq')
'''
 
### Medium query example
#### Prompt from usecase with index: 10, difficulty medium, type object   # tokens:211
medium_ex='''
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
adv_ex='''
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




# Few-Shot example w/ CoT

### Simple query example
#### Prompt from usecase with index: 26, difficulty simple, type object   # tokens:268 
simple_ex1_cot='''
 # Postgres SQL tables, with their properties:
    TABLE probability, columns=[oid, class_name, classifier_name, classifier_version, probability, ranking]
    TABLE object, columns=[oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change]
    # Context:
    ## General information of the schema and the database
    [general_context]
    ## Information useful for the query 
    -- mjd date for August 25 2022 = 59816.0
    -- mjd date for September 10 2022 = 59832.0
    -- object.firstmjd represents the time of the first detection in the given fid in days
    -- object.ndet represents the total number of detections for the object
    REQUEST: Query objects which appeared between august 25 and september 10 of 2022, which are classified as supernova by the stamp classifier with a probability greater than 0.5 or which have more than one detection
    Let's Think Step By Step. The user is asking for general information about objects between two specific dates, information that is in the object table. Also, the user is asking for objects classified by supernova with a specific probability, so we need to use the probability table. 
    Finally, the user is asking for objects with more than one detection, so we can use the object table again to add this condition with an OR statement from the requested probability.
    
    ANSWER: SELECT
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




 
 