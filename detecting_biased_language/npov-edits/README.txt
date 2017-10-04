NPOV edits dataset gotten from: https://www.cs.cornell.edu/~cristian/Biased_language.html

Processed according to clumn 4 = True as specified below and converted to csv via an ipython session (saved) in the same folder.



========================================================================
README file for the NPOV edits

Release: 1.0 (05/03/2013)
Created: May 3, 2013
========================================================================

The NPOV edits accompany the ACL-2013 submission:

Marta Recasens, Cristian Danescu-Niculescu-Mizil, and Dan
Jurafsky. 2013. Linguistic Models for Analyzing and Detecting Biased
Language. Proceedings of ACL 2013.

Please cite this paper if you use this resource in your work.

Contact: recasens@google.com 
	 cristiand@cs.stanford.edu


---- DESCRIPTION ----

The NPOV edits are the collection of edits extracted from the NPOV
corpus, which consists of 7464 Wikipedia articles in the "NPOV
disputes" category together with their history of revisions. The edits
include changes involving strings of up to five words that occurred
between a pair of consecutive revisions.

We keep the train/test/set split that was used for the experiments
presented in the ACL-2013 paper referenced above.


---- FORMAT ----

The edits are released as tab-separated-value (TSV) files. Each line
contains an edit with the following information split across 10
tab-delimited columns: 

1. Title of the Wikipedia article the edit comes from. 
2. Revision number.
3. True if the revision text contains an NPOV tag (e.g., {{POV}}), and
false otherwise.
4. True if the edit comment contains the string "POV", and false
otherwise.
5. ID of the editor responsible for that revision.
6. Size of the revision. It can take three values: minor, major or
unknown (if unspecified).
7. String modified by the edit (i.e., before form).
8. String resulting from the edit (i.e., after form).
9. Original sentence of the before form (i.e., string in column 7).
10. Original sentence of the after form (i.e., string in column 8).

Note: The bias-driven edits discussed in the ACL-2013 paper correspond
to those edits whose column 4 is true.


---- CONTENTS ----

   * 5gram-edits-train.tsv
     Train set containing 1,613,126 edits.

   * 5gram-edits-dev.tsv
     Development set containing 146,641 edits.

   * 5gram-edits-test.tsv
     Test set containing 180,435 edits.

========================================================================
