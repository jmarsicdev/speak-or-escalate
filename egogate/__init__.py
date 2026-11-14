"""egogate: edge VLM with selective cloud escalation for proactive egocentric video.

Pipeline (see configs/default.yaml and README):

    query --> CueParser --> cues (text)
                              |
    video --> Window(2s@2fps) --> Matcher (Ops-MM embeddings, cosine) --> trigger?
                              |                                            |
                              +--> EvidenceExtractor (scene-graph triplets every N s)
                                                                           |
                                                              Decider -> p(speak)
                                                                           |
                                   high: edge answers | mid: escalate to cloud | low: silent
"""

__version__ = "0.1.0"
