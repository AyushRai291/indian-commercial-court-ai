import type {
  AnswerResponse,
  VerificationResponse,
  VerifiedResearchFixture,
} from '../types'

const trfSource =
  'https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=2017/english/2017_7_409_441_EN.pdf'
const perkinsSource =
  'https://api.sci.gov.in/supremecourt/2019/27558/27558_2019_6_1501_18525_Judgement_26-Nov-2019.pdf'
const essarSource =
  'https://api.sci.gov.in/supremecourt/2019/24417/24417_2019_4_1501_18158_Judgement_15-Nov-2019.pdf'

// Static presentation fixtures shaped exactly like the implemented /answer response.
// They are never written to PostgreSQL or presented as a live API response.
export const arbitratorAnswer: AnswerResponse = {
  query: 'Can an ineligible arbitrator nominate another person as arbitrator?',
  answer:
    'Once the named arbitrator becomes ineligible by operation of law, the power attached to that office to nominate another arbitrator is also lost [E1]. The rule extends from an ineligible named arbitrator to an interested decision-maker who exclusively appoints the sole arbitrator [E1][E2]. Section 12(5) permits waiver only through an express written agreement made after disputes have arisen, and that waiver is valid for all future disputes [E3]. The retrieved paragraphs establish that every prior award made under such a clause is automatically void.',
  used_evidence_ids: ['E1', 'E2', 'E3'],
  evidence: [
    {
      evidence_id: 'E1',
      paragraph_uid: 'e20665a8-c3ab-5796-bef7-f0c1f6f632f2',
      case_id: 22,
      case_name: 'TRF Ltd. v. Energo Engineering Projects Ltd.',
      case_number: 'Civil Appeal No. 5306 of 2017',
      court: 'Supreme Court of India',
      judgment_date: '2017-07-03',
      page_number: 33,
      paragraph_number: 84,
      source_url: trfSource,
      text:
        'TRF LTD. v. ENERGO ENGINEERING PROJECTS LTD. 441 [DIPAK MISRA, J.] arbitrator, who may be otherwise eligible and a respectable person. As stated earlier, we are neither concerned with the objectivity nor the individual respectability. We are only concerned with the authority or the power of the Managing Director. By our analysis, we are obligated to arrive at the conclusion that once the arbitrator has become ineligible by operation of law, he cannot nominate another as an arbitrator. The arbitrator becomes ineligible as per prescription contained in Section 12(5) of the Act. It is inconceivable in law that person who is statutorily ineligible can nominate a person. Needless to say, once the infrastructure collapses, the superstructure is bound to collapse. One cannot have a building without the plinth. Or to put it differently, once the identity of the Managing Director as the sole arbitrator is lost, the power to nominate someone else as an arbitrator is obliterated. Therefore, the view expressed by the High Court is not sustainable and we say so.',
      bm25_rank: 1,
      bm25_score: 18.67,
      dense_rank: 2,
      dense_score: 0.84,
      rrf_score: 0.0325,
      hybrid_rank: 1,
      cross_encoder_score: 8.42,
      reranked_rank: 1,
    },
    {
      evidence_id: 'E2',
      paragraph_uid: 'c5594ba7-3d1d-5704-af9e-fa0b11af4ac2',
      case_id: 2,
      case_name: 'Perkins Eastman Architects DPC v. HSCC (India) Ltd.',
      case_number: 'Arbitration Application No. 32 of 2019',
      court: 'Supreme Court of India',
      judgment_date: '2019-11-26',
      page_number: 24,
      paragraph_number: 64,
      source_url: perkinsSource,
      text:
        'Arbitration Application No.32 of 2019 Perkins Eastman Architects DPC & Anr. v. HSCC (India) Ltd. 24 course for dispute resolution. Naturally, the person who has an interest in the outcome or decision of the dispute must not have the power to appoint a sole arbitrator. That has to be taken as the essence of the amendments brought in by the Arbitration and Conciliation (Amendment) Act, 2015 (Act 3 of 2016) and recognised by the decision of this Court in TRF Limited.',
      bm25_rank: 3,
      bm25_score: 16.91,
      dense_rank: 1,
      dense_score: 0.87,
      rrf_score: 0.0323,
      hybrid_rank: 2,
      cross_encoder_score: 7.96,
      reranked_rank: 2,
    },
    {
      evidence_id: 'E3',
      paragraph_uid: 'a5b62777-d935-5458-8c86-8238dfe5fb99',
      case_id: 22,
      case_name: 'TRF Ltd. v. Energo Engineering Projects Ltd.',
      case_number: 'Civil Appeal No. 5306 of 2017',
      court: 'Supreme Court of India',
      judgment_date: '2017-07-03',
      page_number: 12,
      paragraph_number: 5,
      source_url: trfSource,
      text:
        'Notwithstanding any prior agreement to the contrary, any person whose relationship, with the parties or counsel or the subject-matter of the dispute, falls under any of the categories specified in the Seventh Schedule shall be ineligible to be appointed as an arbitrator: Provided that parties may, subsequent to disputes having arisen between them, waive the applicability of this sub-section by an express agreement in writing.',
      bm25_rank: 2,
      bm25_score: 17.24,
      dense_rank: 5,
      dense_score: 0.8,
      rrf_score: 0.0315,
      hybrid_rank: 3,
      cross_encoder_score: 6.71,
      reranked_rank: 3,
    },
  ],
  retrieval_latency_ms: 2894.6,
  generation_latency_ms: 0,
  total_latency_ms: 2894.6,
}

export const commercialWisdomAnswer: AnswerResponse = {
  query:
    'What is the scope of judicial interference with the commercial wisdom of the committee of creditors?',
  answer:
    'Judicial scrutiny of an approved resolution plan is confined to the statutory matters identified in Sections 30(2), 31 and 61(3) of the Insolvency and Bankruptcy Code [E1]. The NCLT and NCLAT cannot replace the commercial decision of financial creditors with their own view or exercise a general equitable jurisdiction [E2]. Review therefore does not extend to reassessing the justness or logic of the creditors’ commercial opinion [E3].',
  used_evidence_ids: ['E1', 'E2', 'E3'],
  evidence: [
    {
      evidence_id: 'E1',
      paragraph_uid: '0c3a93c4-4b64-526a-b230-af44842096eb',
      case_id: 11,
      case_name:
        'Committee of Creditors of Essar Steel India Ltd. v. Satish Kumar Gupta',
      case_number: 'Civil Appeal Nos. 8766–8767 of 2019 and connected matters',
      court: 'Supreme Court of India',
      judgment_date: '2019-11-15',
      page_number: 65,
      paragraph_number: 45,
      source_url: essarSource,
      text:
        'Indubitably, the inquiry in such an appeal would be limited to the power exercisable by the resolution professional under Section 30(2) of the I&B Code or, at best, by the adjudicating authority (NCLT) under Section 31(2) read with 31(1) of the I&B Code. No other inquiry would be permissible. Further, the jurisdiction bestowed upon the appellate authority (NCLAT) is also expressly circumscribed. It can examine the challenge only in relation to the grounds specified in Section 61(3) of the I&B Code, which is limited to matters other than enquiry into the autonomy or commercial wisdom of the dissenting financial creditors. Thus, the prescribed authorities (NCLT/NCLAT) have been endowed with limited jurisdiction as specified in the I&B Code and not to act as a court of equity or exercise plenary powers.',
      bm25_rank: 2,
      bm25_score: 20.14,
      dense_rank: 1,
      dense_score: 0.89,
      rrf_score: 0.0325,
      hybrid_rank: 1,
      cross_encoder_score: 8.03,
      reranked_rank: 1,
    },
    {
      evidence_id: 'E2',
      paragraph_uid: 'd6f64d61-2a0c-5f51-96f7-f7fb7154d701',
      case_id: 11,
      case_name:
        'Committee of Creditors of Essar Steel India Ltd. v. Satish Kumar Gupta',
      case_number: 'Civil Appeal Nos. 8766–8767 of 2019 and connected matters',
      court: 'Supreme Court of India',
      judgment_date: '2019-11-15',
      page_number: 65,
      paragraph_number: 46,
      source_url: essarSource,
      text:
        'In our view, neither the adjudicating authority (NCLT) nor the appellate authority (NCLAT) has been endowed with the jurisdiction to reverse the commercial wisdom of the dissenting financial creditors and that too on the specious ground that it is only an opinion of the minority.',
      bm25_rank: 1,
      bm25_score: 21.03,
      dense_rank: 4,
      dense_score: 0.83,
      rrf_score: 0.032,
      hybrid_rank: 2,
      cross_encoder_score: 7.58,
      reranked_rank: 2,
    },
    {
      evidence_id: 'E3',
      paragraph_uid: '903eda94-04ed-5e46-8177-ad9d00203a5e',
      case_id: 11,
      case_name:
        'Committee of Creditors of Essar Steel India Ltd. v. Satish Kumar Gupta',
      case_number: 'Civil Appeal Nos. 8766–8767 of 2019 and connected matters',
      court: 'Supreme Court of India',
      judgment_date: '2019-11-15',
      page_number: 67,
      paragraph_number: 51,
      source_url: essarSource,
      text:
        'At best, the Adjudicating Authority (NCLT) may cause an enquiry into the approved resolution plan on limited grounds referred to in Section 30(2) read with Section 31(1) of the I&B Code. It cannot make any other inquiry nor is competent to issue any direction in relation to the exercise of commercial wisdom of the financial creditors — be it for approving, rejecting or abstaining, as the case may be. Even the inquiry before the Appellate Authority (NCLAT) is limited to the grounds under Section 61(3) of the I&B Code. It does not postulate jurisdiction to undertake scrutiny of the justness of the opinion expressed by financial creditors at the time of voting.',
      bm25_rank: 5,
      bm25_score: 17.82,
      dense_rank: 2,
      dense_score: 0.86,
      rrf_score: 0.0318,
      hybrid_rank: 3,
      cross_encoder_score: 7.11,
      reranked_rank: 3,
    },
  ],
  retrieval_latency_ms: 2910.1,
  generation_latency_ms: 0,
  total_latency_ms: 2910.1,
}

export const arbitratorVerification: VerificationResponse = {
  claims: [
    {
      claim_id: 'C1',
      claim:
        'Once the named arbitrator becomes ineligible by operation of law, the power attached to that office to nominate another arbitrator is also lost.',
      citation_ids: ['E1'],
      status: 'SUPPORTED',
      reason:
        'E1 directly states that statutory ineligibility removes the managing director\'s power to nominate another arbitrator.',
      evidence_uids: ['e20665a8-c3ab-5796-bef7-f0c1f6f632f2'],
    },
    {
      claim_id: 'C2',
      claim:
        'The rule extends from an ineligible named arbitrator to an interested decision-maker who exclusively appoints the sole arbitrator.',
      citation_ids: ['E1', 'E2'],
      status: 'SUPPORTED',
      reason:
        'E1 removes the ineligible nominee\'s appointment power, while E2 expressly bars an interested person from appointing the sole arbitrator.',
      evidence_uids: [
        'e20665a8-c3ab-5796-bef7-f0c1f6f632f2',
        'c5594ba7-3d1d-5704-af9e-fa0b11af4ac2',
      ],
    },
    {
      claim_id: 'C3',
      claim:
        'Section 12(5) permits waiver only through an express written agreement made after disputes have arisen, and that waiver is valid for all future disputes.',
      citation_ids: ['E3'],
      status: 'PARTIAL',
      reason:
        'E3 supports an express written waiver after a dispute arises, but does not say that one waiver governs all future disputes.',
      evidence_uids: ['a5b62777-d935-5458-8c86-8238dfe5fb99'],
    },
    {
      claim_id: 'C4',
      claim:
        'The retrieved paragraphs establish that every prior award made under such a clause is automatically void.',
      citation_ids: [],
      status: 'UNSUPPORTED',
      reason: 'No evidence citation was attached to this material claim.',
      evidence_uids: [],
    },
  ],
  summary: {
    supported: 2,
    partial: 1,
    unsupported: 1,
  },
  claim_extraction_latency_ms: 1.8,
  verification_latency_ms: 1264.4,
  total_latency_ms: 1266.2,
}

export const commercialWisdomVerification: VerificationResponse = {
  claims: [
    {
      claim_id: 'C1',
      claim:
        'Judicial scrutiny of an approved resolution plan is confined to the statutory matters identified in Sections 30(2), 31 and 61(3) of the Insolvency and Bankruptcy Code.',
      citation_ids: ['E1'],
      status: 'SUPPORTED',
      reason:
        'E1 directly limits NCLT and NCLAT review to the identified statutory grounds.',
      evidence_uids: ['0c3a93c4-4b64-526a-b230-af44842096eb'],
    },
    {
      claim_id: 'C2',
      claim:
        'The NCLT and NCLAT cannot replace the commercial decision of financial creditors with their own view or exercise a general equitable jurisdiction.',
      citation_ids: ['E1', 'E2'],
      status: 'SUPPORTED',
      reason:
        'E1 excludes equity or plenary review and E2 bars reversal of creditors\' commercial wisdom.',
      evidence_uids: [
        '0c3a93c4-4b64-526a-b230-af44842096eb',
        'd6f64d61-2a0c-5f51-96f7-f7fb7154d701',
      ],
    },
    {
      claim_id: 'C3',
      claim:
        'Review therefore does not extend to reassessing the justness or logic of the creditors\' commercial opinion.',
      citation_ids: ['E3'],
      status: 'SUPPORTED',
      reason:
        'E3 expressly says the appellate inquiry does not include scrutiny of the justness of creditors\' opinion.',
      evidence_uids: ['903eda94-04ed-5e46-8177-ad9d00203a5e'],
    },
  ],
  summary: {
    supported: 3,
    partial: 0,
    unsupported: 0,
  },
  claim_extraction_latency_ms: 1.3,
  verification_latency_ms: 1187.7,
  total_latency_ms: 1189,
}

export const arbitratorResearch: VerifiedResearchFixture = {
  answer: arbitratorAnswer,
  verification: arbitratorVerification,
}

export const commercialWisdomResearch: VerifiedResearchFixture = {
  answer: commercialWisdomAnswer,
  verification: commercialWisdomVerification,
}

export const mockResearchResults = [arbitratorResearch, commercialWisdomResearch]
