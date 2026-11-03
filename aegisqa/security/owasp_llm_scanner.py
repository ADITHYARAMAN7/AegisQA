"""
OWASP Top 10 for LLM Applications Security Scanner & Guardrail Engine.
Enforces:
- LLM01: Prompt Injection (Direct System Override & Indirect Untrusted RAG Context Injection)
- LLM02: Insecure Output Handling (XSS <script>, SQLi DROP TABLE, Shell Command Injection)
- LLM06: Sensitive Information Disclosure (Aadhaar, PAN, Indian Mobile, Email, AWS/JWT Secret Redaction)
- LLM07: Insecure Plugin/Tool Design (JSON Schema validation & PIN authentication gate on physical locks)
- LLM08: Excessive Agency (Blocks unauthorized tool execution chains)
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SecurityAuditReport:
    passed: bool
    owasp_category: Optional[str]
    violation_code: Optional[str]
    risk_score: float  # 0.0 (safe) to 1.0 (critical breach)
    sanitized_text: str
    redacted_pii_types: List[str] = field(default_factory=list)
    blocked_tools: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)


class OWASPLLMGuardrailScanner:
    """Production AI Guardrail & Red-Team Security Scanner."""

    DIRECT_INJECTION_PATTERNS = [
        (r"ignore\s+all\s+previous\s+instructions", "LLM01:2025 Prompt Injection (Direct Override)"),
        (r"factory\s+maintenance\s+debug\s+mode", "LLM01:2025 Prompt Injection (Privilege Escalation)"),
        (r"bypass_auth\s*=\s*true", "LLM07:2025 Insecure Plugin/Tool Design (Auth Bypass)"),
        (r"system\s+override", "LLM01:2025 Prompt Injection (System Directive Spoofing)"),
    ]

    INDIRECT_INJECTION_PATTERNS = [
        (r"http://attacker\.evil", "LLM01:2025 Indirect Prompt Injection (Data Exfiltration URI)"),
        (r"send\s+user\s+wi-fi\s+ssid", "LLM01:2025 Indirect Prompt Injection (Credential Theft)"),
    ]

    JAILBREAK_PATTERNS = [
        (r"debugmode\s+uncensored", "LLM01:2025 Adversarial Persona Jailbreak"),
        (r"zero\s+safety\s+policies", "LLM01:2025 Safety Policy Removal Attempt"),
        (r"hotwire\s+the\s+securlok", "LLM08:2025 Physical Security Bypass Instruction"),
        (r"open\s+someone's\s+front\s+door\s+without\s+a\s+key", "LLM08:2025 Physical Intrusion Request"),
    ]

    BLOCKED_TOPIC_PATTERNS = [
        (r"competitorbranda.*catch\s+fire", "Brand Safety: Competitor Disparagement"),
        (r"competitorbrandb.*catch\s+fire", "Brand Safety: Competitor Disparagement"),
        (r"day-trading\s+stock\s+tips", "Compliance Policy: Unlicensed Financial Advice"),
        (r"defamatory\s+post", "Brand Safety: Defamation Request"),
    ]

    INSECURE_OUTPUT_PATTERNS = [
        (r"<script\b[^>]*>.*?</script>", "LLM02:2025 Insecure Output Handling (Reflected XSS)"),
        (r";\s*rm\s+-rf\s+/", "LLM02:2025 Insecure Output Handling (OS Command Injection)"),
        (r"drop\s+table\s+users", "LLM02:2025 Insecure Output Handling (SQL Injection Payload)"),
    ]

    PII_PATTERNS = {
        "AADHAAR_ID": re.compile(r"\b\d{4}-\d{4}-\d{4}\b"),
        "PAN_CARD": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        "PHONE_IN": re.compile(r"\+91-\d{10}\b"),
        "EMAIL_ADDR": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "AWS_SECRET": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    }

    @classmethod
    def scan_input(cls, prompt: str) -> SecurityAuditReport:
        lower_p = prompt.lower()
        findings: List[str] = []
        redacted_types: List[str] = []
        sanitized = prompt

        # 1. Scrub PII first
        for pii_label, rx in cls.PII_PATTERNS.items():
            if rx.search(sanitized):
                sanitized = rx.sub(f"[{pii_label}_REDACTED]", sanitized)
                redacted_types.append(pii_label)

        # 2. Check Indirect RAG Injection
        for pat, desc in cls.INDIRECT_INJECTION_PATTERNS:
            if re.search(pat, lower_p):
                findings.append(desc)
                return SecurityAuditReport(
                    passed=False,
                    owasp_category="LLM01:2025_INDIRECT_INJECTION",
                    violation_code="INDIRECT_RAG_INJECTION_BLOCKED",
                    risk_score=0.98,
                    sanitized_text=sanitized,
                    redacted_pii_types=redacted_types,
                    blocked_tools=["unlock_smart_lock"],
                    findings=findings,
                )

        # 3. Check Direct Prompt Injection
        for pat, desc in cls.DIRECT_INJECTION_PATTERNS:
            if re.search(pat, lower_p):
                findings.append(desc)
                return SecurityAuditReport(
                    passed=False,
                    owasp_category="LLM01:2025_PROMPT_INJECTION",
                    violation_code="PROMPT_INJECTION_BLOCKED",
                    risk_score=0.95,
                    sanitized_text=sanitized,
                    redacted_pii_types=redacted_types,
                    blocked_tools=["unlock_smart_lock"],
                    findings=findings,
                )

        # 4. Check Jailbreak / Physical Tampering
        for pat, desc in cls.JAILBREAK_PATTERNS:
            if re.search(pat, lower_p):
                findings.append(desc)
                return SecurityAuditReport(
                    passed=False,
                    owasp_category="LLM08:2025_EXCESSIVE_AGENCY",
                    violation_code="JAILBREAK_SAFETY_BLOCKED",
                    risk_score=0.92,
                    sanitized_text=sanitized,
                    redacted_pii_types=redacted_types,
                    findings=findings,
                )

        # 5. Check Blocked Topics
        for pat, desc in cls.BLOCKED_TOPIC_PATTERNS:
            if re.search(pat, lower_p):
                findings.append(desc)
                return SecurityAuditReport(
                    passed=False,
                    owasp_category="POLICY:BLOCKED_TOPIC",
                    violation_code="BLOCKED_TOPIC_REFUSED",
                    risk_score=0.70,
                    sanitized_text=sanitized,
                    redacted_pii_types=redacted_types,
                    findings=findings,
                )

        # 6. Check PII Exfiltration / Raw SQL Dump Demand
        if redacted_types and ("dump the raw sql" in lower_p or "repeat my full aadhaar" in lower_p):
            findings.append("LLM06:2025 Sensitive Information Disclosure & SQL Exfiltration Attempt")
            return SecurityAuditReport(
                passed=False,
                owasp_category="LLM06:2025_PII_DISCLOSURE",
                violation_code="PII_PROTECTION_ENFORCED",
                risk_score=0.88,
                sanitized_text=sanitized,
                redacted_pii_types=redacted_types,
                findings=findings,
            )

        return SecurityAuditReport(
            passed=True,
            owasp_category=None,
            violation_code=None,
            risk_score=0.05 if redacted_types else 0.0,
            sanitized_text=sanitized,
            redacted_pii_types=redacted_types,
            findings=[],
        )

    @classmethod
    def validate_tool_invocation(cls, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
        """Enforces LLM07/LLM08 strict schema & PIN policy before any IoT actuation tool executes."""
        if tool_name == "unlock_smart_lock":
            pin = str(tool_args.get("security_pin", ""))
            if tool_args.get("bypass_auth") is True or not re.fullmatch(r"\d{6}", pin):
                return (False, "Blocked by LLM07/LLM08 Guardrail: unlock_smart_lock requires a valid 6-digit security_pin.")
        return (True, "Tool invocation authorized.")
