"""Generate train/val MCQ datasets from the Nexus Cybersecurity Framework.

Each question tests a specific fact from the policy. Distractors are plausible
but incorrect values. Train and val sets test different facts (no overlap).
"""

import json
import random
from pathlib import Path

ANSWER_INSTRUCTION = "\nPlease reason step by step."


def make_q(question: str, options: list[str], correct_idx: int) -> dict:
    """Build an MCQ dict. Options are stored with the correct_idx for later shuffling."""
    return {
        "_question": question,
        "_options": options,
        "_correct_idx": correct_idx,
    }


def finalize_q(raw: dict, rng: random.Random) -> dict:
    """Shuffle options randomly and format the final MCQ dict."""
    letters = ["A", "B", "C", "D"]
    options = list(raw["_options"])
    correct_text = options[raw["_correct_idx"]]

    rng.shuffle(options)
    new_correct_idx = options.index(correct_text)

    option_lines = "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))
    prompt = f"{raw['_question']}\n\n{option_lines}{ANSWER_INSTRUCTION}"
    return {
        "prompt": prompt,
        "answer": letters[new_correct_idx],
        "kind": "mcq",
        "system": "Given a question and four options, please select the right answer. Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\nFor the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else.",
    }


# === TRAINING QUESTIONS ===
# Each tests a distinct, specific fact from the policy.

TRAIN_QUESTIONS = [
    # --- Section 1: Identity & Access Management ---
    make_q(
        "What is the minimum password length for administrator accounts?",
        ["16 characters", "18 characters", "22 characters", "24 characters"],
        2,
    ),
    make_q(
        "How often must administrator passwords be rotated?",
        ["Every 30 days", "Every 45 days", "Every 60 days", "Every 90 days"],
        1,
    ),
    make_q(
        "What is the minimum password length for standard user accounts?",
        ["10 characters", "12 characters", "14 characters", "16 characters"],
        2,
    ),
    make_q(
        "How often must standard user passwords be rotated?",
        ["Every 30 days", "Every 60 days", "Every 90 days", "Every 120 days"],
        2,
    ),
    make_q(
        "What is the minimum password length for service accounts?",
        ["22 characters", "24 characters", "28 characters", "32 characters"],
        3,
    ),
    make_q(
        "How often must service account passwords be rotated?",
        ["Every 14 days", "Every 30 days", "Every 45 days", "Every 60 days"],
        1,
    ),
    make_q(
        "What is the minimum password length for contractor accounts?",
        ["12 characters", "14 characters", "16 characters", "18 characters"],
        2,
    ),
    make_q(
        "How often must contractor account passwords be rotated?",
        ["Every 30 days", "Every 45 days", "Every 60 days", "Every 90 days"],
        2,
    ),
    make_q(
        "At which data classification level does MFA become mandatory?",
        ["C1 and above", "C2 and above", "C3 and above", "All levels"],
        1,
    ),
    make_q(
        "Which MFA method is explicitly prohibited by the NCF?",
        ["Hardware tokens", "TOTP authenticator apps", "SMS-based MFA", "Biometric authentication"],
        2,
    ),
    make_q(
        "Within how many hours must MFA enrollment be completed after account creation?",
        ["24 hours", "48 hours", "72 hours", "96 hours"],
        2,
    ),
    make_q(
        "After how many failed login attempts is a standard user account locked out?",
        ["3 attempts", "5 attempts", "7 attempts", "10 attempts"],
        1,
    ),
    make_q(
        "What is the lockout duration for administrator accounts?",
        ["30 minutes", "1 hour", "2 hours", "4 hours"],
        3,
    ),
    make_q(
        "After how many failed login attempts is an administrator account locked out?",
        ["3 attempts", "5 attempts", "7 attempts", "10 attempts"],
        0,
    ),
    make_q(
        "What happens when a service account exceeds the failed login threshold?",
        ["Account is locked for 30 minutes", "Account is disabled permanently", "Alert is sent to SOC immediately", "Account is locked for 4 hours"],
        2,
    ),
    make_q(
        "What is the inactivity timeout for administrative consoles?",
        ["5 minutes", "10 minutes", "15 minutes", "20 minutes"],
        2,
    ),
    make_q(
        "What is the maximum lifetime for API tokens?",
        ["4 hours", "8 hours", "12 hours", "24 hours"],
        1,
    ),
    make_q(
        "How often are VPN sessions required to re-authenticate?",
        ["Every 4 hours", "Every 8 hours", "Every 12 hours", "Every 24 hours"],
        2,
    ),
    make_q(
        "How frequently are privileged access reviews conducted?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        1,
    ),
    make_q(
        "What is the SLA for provisioning a new user account?",
        ["24 hours", "48 hours", "72 hours", "96 hours"],
        1,
    ),
    make_q(
        "Within how many hours must an account be deprovisioned after employee termination?",
        ["1 hour", "4 hours", "8 hours", "24 hours"],
        1,
    ),
    make_q(
        "How often are standard access reviews conducted?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        2,
    ),
    make_q(
        "How often are service account reviews conducted?",
        ["Weekly", "Biweekly", "Monthly", "Quarterly"],
        2,
    ),
    make_q(
        "What is the lockout duration for standard user accounts?",
        ["15 minutes", "30 minutes", "1 hour", "2 hours"],
        1,
    ),
    make_q(
        "After how many failed login attempts is a contractor account locked out?",
        ["3 attempts", "4 attempts", "5 attempts", "7 attempts"],
        1,
    ),
    make_q(
        "What is the lockout duration for contractor accounts?",
        ["30 minutes", "1 hour", "2 hours", "4 hours"],
        2,
    ),
    make_q(
        "What is the inactivity timeout for standard workstations?",
        ["15 minutes", "20 minutes", "30 minutes", "45 minutes"],
        2,
    ),
    make_q(
        "After how many failed login attempts is a service account locked?",
        ["5 attempts", "7 attempts", "10 attempts", "15 attempts"],
        2,
    ),
    # --- Section 2: Data Classification ---
    make_q(
        "What classification level is described as 'freely shareable information'?",
        ["C0 (Public)", "C1 (Internal)", "C2 (Confidential)", "C3 (Restricted)"],
        0,
    ),
    make_q(
        "What classification level corresponds to 'Restricted'?",
        ["C1", "C2", "C3", "C4"],
        2,
    ),
    make_q(
        "What encryption standard is required for C3 and C4 data at rest?",
        ["AES-128-GCM", "AES-256-CBC", "AES-256-GCM", "ChaCha20-Poly1305"],
        2,
    ),
    make_q(
        "What is the minimum encryption for C2 data at rest?",
        ["AES-128-CBC", "AES-128-GCM", "AES-256-GCM", "No encryption required"],
        1,
    ),
    make_q(
        "What special encryption requirement applies to C4 data in transit?",
        ["TLS 1.3 only", "Double encryption (TLS 1.3 + application-layer AES-256)", "Triple DES encryption", "IPSec with AES-256"],
        1,
    ),
    make_q(
        "Starting at which classification level must all access be logged?",
        ["C1 and above", "C2 and above", "C3 and above", "All levels"],
        1,
    ),
    make_q(
        "What access rule applies specifically to C4 classified data?",
        ["Manager approval required", "VPN access only", "Dual-person access rule", "Read-only access"],
        2,
    ),
    make_q(
        "What is the maximum retention period for C1 (Internal) data?",
        ["3 years", "5 years", "7 years", "Indefinite"],
        2,
    ),
    make_q(
        "What is the maximum retention period for C3 (Restricted) data?",
        ["1 year", "3 years", "5 years", "7 years"],
        1,
    ),
    make_q(
        "What is the maximum retention period for C4 (Top Secret) data?",
        ["6 months", "1 year", "2 years", "3 years"],
        1,
    ),
    make_q(
        "What destruction method is required for C4 data after retention?",
        ["Cryptographic erasure", "Secure deletion software", "Physical destruction of storage media", "Degaussing only"],
        2,
    ),
    make_q(
        "Who must approve cross-border transfer of C3 or C4 data?",
        ["Department head", "CTO", "CISO", "CEO"],
        2,
    ),
    make_q(
        "Which datacenters comprise 'Region Alpha' for C4 data residency?",
        ["US-East-1 and US-West-1", "US-East-1 and EU-West-1", "EU-West-1 and AP-Southeast-1", "US-East-1 and AP-Northeast-1"],
        1,
    ),
    make_q(
        "What TLS version is required for C3 data in transit?",
        ["TLS 1.0 or above", "TLS 1.1 or above", "TLS 1.2 or above", "TLS 1.3 mandatory"],
        3,
    ),
    make_q(
        "What is the maximum retention period for C2 (Confidential) data?",
        ["3 years", "5 years", "7 years", "10 years"],
        1,
    ),
    make_q(
        "What destruction method is required for C2 data?",
        ["Standard deletion", "Cryptographic erasure", "Physical destruction", "Overwrite three times"],
        1,
    ),
    make_q(
        "Where must C4 data be stored physically?",
        ["Any secure server room", "Zone C facilities", "Zone D air-gapped systems", "Encrypted cloud storage"],
        2,
    ),
    make_q(
        "Who must approve cross-border transfer of C2 data?",
        ["CISO", "Department head", "CTO", "No approval needed"],
        1,
    ),
    make_q(
        "Can C0 data be retained indefinitely?",
        ["No, maximum 1 year", "No, maximum 5 years", "No, maximum 7 years", "Yes, indefinite retention permitted"],
        3,
    ),
    make_q(
        "What additional requirement exists for C3 data access beyond logging?",
        ["VPN only", "Manager approval and need-to-know justification", "Hardware token MFA", "Air-gapped terminal only"],
        1,
    ),
    # --- Section 3: Network Security ---
    make_q(
        "Which network segment is designated for C4 systems?",
        ["GREEN", "BLUE", "AMBER", "RED"],
        3,
    ),
    make_q(
        "Which network segment is air-gapped with no internet connectivity?",
        ["AMBER", "RED", "BLUE", "All segments above GREEN"],
        1,
    ),
    make_q(
        "How often are RED segment firewall rules reviewed?",
        ["Daily", "Weekly", "Biweekly", "Monthly"],
        1,
    ),
    make_q(
        "How often are AMBER segment firewall rules reviewed?",
        ["Weekly", "Biweekly", "Monthly", "Quarterly"],
        1,
    ),
    make_q(
        "How often are GREEN segment firewall rules reviewed?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        1,
    ),
    make_q(
        "How frequently is the IDS signature database updated?",
        ["Every hour", "Every 4 hours", "Every 8 hours", "Daily"],
        1,
    ),
    make_q(
        "How often are penetration tests conducted on the RED segment?",
        ["Weekly", "Monthly", "Quarterly", "Semi-annually"],
        1,
    ),
    make_q(
        "How often are penetration tests conducted on the AMBER segment?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        1,
    ),
    make_q(
        "How often are penetration tests conducted on BLUE and GREEN segments?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        3,
    ),
    make_q(
        "Which firm conducts external penetration testing for Nexus?",
        ["CrowdStrike", "Meridian Assurance Partners", "Sentinel Cyber Assurance", "Mandiant"],
        2,
    ),
    make_q(
        "What VPN protocol does the NCF mandate?",
        ["OpenVPN", "IPSec/IKEv2", "WireGuard", "L2TP/IPSec"],
        2,
    ),
    make_q(
        "What is the VPN re-keying interval?",
        ["30 minutes", "60 minutes", "90 minutes", "120 minutes"],
        2,
    ),
    make_q(
        "Is split tunneling permitted for C2+ access?",
        ["Yes, always", "Yes, with manager approval", "No, prohibited", "Only on corporate devices"],
        2,
    ),
    make_q(
        "What is the maximum number of concurrent VPN sessions per user?",
        ["1", "2", "3", "5"],
        1,
    ),
    make_q(
        "How long are VPN access logs retained?",
        ["6 months", "1 year", "2 years", "5 years"],
        1,
    ),
    make_q(
        "What is the maximum VLAN size according to the NCF?",
        ["100 hosts", "150 hosts", "250 hosts", "500 hosts"],
        2,
    ),
    make_q(
        "What authentication standard is required on all wired ports?",
        ["RADIUS", "IEEE 802.1X", "TACACS+", "Kerberos"],
        1,
    ),
    make_q(
        "What wireless security standard is required for corporate access?",
        ["WPA2-Enterprise", "WPA2-Personal", "WPA3-Enterprise", "WPA3-Personal"],
        2,
    ),
    make_q(
        "What is the corporate wireless SSID?",
        ["NexusCorp", "NexusSecure", "NexusNet", "NexusWireless"],
        1,
    ),
    make_q(
        "Within how many days must critical penetration test findings be remediated?",
        ["7 days", "14 days", "30 days", "60 days"],
        2,
    ),
    make_q(
        "How often are BLUE segment firewall rules reviewed?",
        ["Weekly", "Biweekly", "Monthly", "Quarterly"],
        2,
    ),
    make_q(
        "On which segments is DNS filtering logging enabled?",
        ["All segments", "AMBER and above", "RED only", "BLUE and above"],
        1,
    ),
    make_q(
        "Who must approve all firewall rule changes?",
        ["CISO", "CTO", "Network Security Lead", "SOC Manager"],
        2,
    ),
    make_q(
        "What is the name of the guest wireless SSID?",
        ["NexusPublic", "NexusGuest", "GuestWifi", "NexusVisitor"],
        1,
    ),
    # --- Section 4: Incident Response ---
    make_q(
        "What is the initial response time requirement for SEV-1 incidents?",
        ["5 minutes", "10 minutes", "15 minutes", "30 minutes"],
        2,
    ),
    make_q(
        "Within how many minutes must the CISO be notified of a SEV-1 incident?",
        ["15 minutes", "30 minutes", "60 minutes", "120 minutes"],
        1,
    ),
    make_q(
        "What is the initial response time requirement for SEV-2 incidents?",
        ["15 minutes", "30 minutes", "1 hour", "2 hours"],
        2,
    ),
    make_q(
        "What is the escalation timeframe for SEV-2 incidents?",
        ["1 hour", "2 hours", "4 hours", "8 hours"],
        2,
    ),
    make_q(
        "What is the initial response time for SEV-3 incidents?",
        ["1 hour", "2 hours", "4 hours", "8 hours"],
        2,
    ),
    make_q(
        "What is the initial response time for SEV-4 incidents?",
        ["4 hours", "8 hours", "12 hours", "24 hours"],
        3,
    ),
    make_q(
        "How often is the incident commander rotated during active SEV-1/SEV-2 incidents?",
        ["Every 4 hours", "Every 6 hours", "Every 8 hours", "Every 12 hours"],
        2,
    ),
    make_q(
        "Within how many minutes must a communication bridge be established for SEV-1?",
        ["5 minutes", "10 minutes", "15 minutes", "30 minutes"],
        1,
    ),
    make_q(
        "Who handles external communication during SEV-1 and SEV-2 incidents?",
        ["CISO directly", "Incident commander", "Designated Public Relations team", "Legal department"],
        2,
    ),
    make_q(
        "Within how many hours must forensic imaging be completed after SEV-1 detection?",
        ["1 hour", "2 hours", "4 hours", "8 hours"],
        1,
    ),
    make_q(
        "Within how many hours must a post-incident review be conducted for SEV-1?",
        ["24 hours", "48 hours", "72 hours", "1 week"],
        2,
    ),
    make_q(
        "Within what timeframe must a PIR be completed for SEV-3 and SEV-4?",
        ["72 hours", "1 week", "2 weeks", "1 month"],
        2,
    ),
    make_q(
        "What is the minimum evidence retention period for incidents?",
        ["6 months", "12 months", "18 months", "24 months"],
        2,
    ),
    make_q(
        "Within how many business days must lessons learned be distributed?",
        ["2 days", "3 days", "5 days", "10 days"],
        2,
    ),
    make_q(
        "How often are tabletop exercises conducted for SEV-1 scenarios?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        1,
    ),
    make_q(
        "Which severity level is described as 'active data breach, ransomware, nation-state attack'?",
        ["SEV-1", "SEV-2", "SEV-3", "SEV-4"],
        0,
    ),
    make_q(
        "Which severity level includes 'compromised admin account' and 'DDoS'?",
        ["SEV-1", "SEV-2", "SEV-3", "SEV-4"],
        1,
    ),
    make_q(
        "What is the escalation timeframe for SEV-3 incidents?",
        ["4 hours", "8 hours", "24 hours", "48 hours"],
        2,
    ),
    # --- Section 5: Cryptographic Standards ---
    make_q(
        "What symmetric encryption standard is required for data at rest?",
        ["AES-128-GCM", "AES-256-CBC", "AES-256-GCM", "ChaCha20-Poly1305"],
        2,
    ),
    make_q(
        "What encryption algorithm is specified for mobile devices?",
        ["AES-128-GCM", "AES-256-GCM", "ChaCha20-Poly1305", "Twofish-256"],
        2,
    ),
    make_q(
        "Which block cipher mode has been prohibited since January 2024?",
        ["GCM", "CTR", "CBC", "ECB"],
        2,
    ),
    make_q(
        "What asymmetric algorithm is required for new implementations of digital signatures?",
        ["RSA-2048", "RSA-4096", "Ed25519", "ECDSA P-256"],
        2,
    ),
    make_q(
        "Until what date are RSA-4096 keys permitted for legacy systems?",
        ["June 2025", "December 2025", "March 2026", "December 2026"],
        1,
    ),
    make_q(
        "How often must RSA keys be rotated?",
        ["Every year", "Every 2 years", "Every 3 years", "Every 5 years"],
        1,
    ),
    make_q(
        "How often must Ed25519 keys be rotated?",
        ["Every year", "Every 2 years", "Every 3 years", "Every 5 years"],
        2,
    ),
    make_q(
        "What is the minimum hashing standard?",
        ["SHA-256", "SHA-384", "SHA-512", "BLAKE3"],
        1,
    ),
    make_q(
        "What is the preferred hashing algorithm?",
        ["SHA-384", "SHA-512", "SHA-3-256", "BLAKE3"],
        3,
    ),
    make_q(
        "What password hashing algorithm is required?",
        ["bcrypt", "scrypt", "Argon2id", "PBKDF2"],
        2,
    ),
    make_q(
        "What is the minimum memory parameter for Argon2id password hashing?",
        ["32 MB", "64 MB", "128 MB", "256 MB"],
        1,
    ),
    make_q(
        "What is the name of the internal Certificate Authority?",
        ["Nexus TrustRoot G2", "Nexus TrustRoot G3", "Nexus CA Root", "Nexus PKI Primary"],
        1,
    ),
    make_q(
        "What is the maximum lifetime for server certificates?",
        ["90 days", "365 days", "398 days", "730 days"],
        2,
    ),
    make_q(
        "What is the maximum lifetime for code signing certificates?",
        ["365 days", "398 days", "730 days", "1095 days"],
        2,
    ),
    make_q(
        "What is the minimum TLS version allowed?",
        ["TLS 1.0", "TLS 1.1", "TLS 1.2", "TLS 1.3"],
        3,
    ),
    make_q(
        "Until what date is TLS 1.2 permitted for legacy integrations?",
        ["June 2025", "December 2025", "March 2026", "June 2026"],
        1,
    ),
    make_q(
        "What is the minimum HMAC standard for message authentication?",
        ["HMAC-SHA-256", "HMAC-SHA-384", "HMAC-SHA-512", "HMAC-BLAKE3"],
        1,
    ),
    make_q(
        "Within how many hours must certificate revocation be completed after compromise?",
        ["4 hours", "12 hours", "24 hours", "48 hours"],
        2,
    ),
    make_q(
        "What is the maximum lifetime for client certificates?",
        ["90 days", "180 days", "365 days", "398 days"],
        2,
    ),
    make_q(
        "What key exchange algorithm is required for new implementations?",
        ["RSA key exchange", "ECDH P-256", "X25519", "DH-2048"],
        2,
    ),
    make_q(
        "Is perfect forward secrecy required for TLS connections?",
        ["Only for C3+ systems", "Only for external connections", "Yes, mandatory for all", "No, recommended only"],
        2,
    ),
    make_q(
        "What Argon2id parallelism parameter is required?",
        ["1", "2", "4", "8"],
        2,
    ),
    make_q(
        "How many iterations are required for Argon2id password hashing?",
        ["1 iteration", "2 iterations", "3 iterations", "5 iterations"],
        2,
    ),
    make_q(
        "Until when is SHA-256 permitted under the grandfather clause?",
        ["March 2025", "June 2025", "December 2025", "March 2026"],
        1,
    ),
    # --- Section 6: Endpoint Security ---
    make_q(
        "What is the EDR agent check-in interval?",
        ["1 minute", "5 minutes", "10 minutes", "15 minutes"],
        1,
    ),
    make_q(
        "What is the name of the approved EDR platform?",
        ["Nexus Defender v3", "Nexus Shield v4", "Nexus Guard Pro", "Nexus Sentinel v5"],
        1,
    ),
    make_q(
        "What is the patching SLA for critical vulnerabilities?",
        ["4 hours", "12 hours", "24 hours", "48 hours"],
        2,
    ),
    make_q(
        "What is the patching SLA for high severity vulnerabilities?",
        ["24 hours", "48 hours", "72 hours", "7 days"],
        2,
    ),
    make_q(
        "What is the patching SLA for medium vulnerabilities?",
        ["3 days", "7 days", "14 days", "21 days"],
        2,
    ),
    make_q(
        "What is the patching SLA for low severity vulnerabilities?",
        ["14 days", "21 days", "30 days", "60 days"],
        2,
    ),
    make_q(
        "What full disk encryption is required on macOS?",
        ["BitLocker", "FileVault", "LUKS", "VeraCrypt"],
        1,
    ),
    make_q(
        "What full disk encryption is required on Linux systems?",
        ["BitLocker", "FileVault", "LUKS", "dm-crypt only"],
        2,
    ),
    make_q(
        "What approvals are required to enable USB storage on an endpoint?",
        ["Manager approval only", "Security team approval only", "Both manager and security team approval", "CISO approval"],
        2,
    ),
    make_q(
        "For which classification level is browser isolation mandatory?",
        ["C2 and above", "C3 and above", "C4 only", "All levels"],
        1,
    ),
    make_q(
        "How often is automated vulnerability scanning performed on servers?",
        ["Daily", "Weekly", "Biweekly", "Monthly"],
        1,
    ),
    make_q(
        "How often is vulnerability scanning performed on internet-facing assets?",
        ["Hourly", "Daily", "Weekly", "Biweekly"],
        1,
    ),
    make_q(
        "What is the minimum PIN length for the BYOD work container?",
        ["4 digits", "6 digits", "8 digits", "Biometric only"],
        1,
    ),
    make_q(
        "What is the name of the approved enterprise app store?",
        ["Nexus Store", "Nexus AppVault", "Nexus Market", "Nexus Apps"],
        1,
    ),
    make_q(
        "What happens if a jailbroken or rooted device is detected?",
        ["User receives a warning", "Device is automatically blocked", "Device is wiped remotely", "Nothing, user is notified"],
        1,
    ),
    make_q(
        "What is the name of the MDM platform for BYOD enrollment?",
        ["Nexus Device Manager", "Nexus Mobile Shield", "Nexus MDM Pro", "Nexus Endpoint Guard"],
        1,
    ),
    make_q(
        "What happens when EDR agent tampering is detected?",
        ["Alert only", "Auto-quarantine and SOC alert", "Device shutdown", "Network isolation only"],
        1,
    ),
    make_q(
        "Who can authorize emergency out-of-band patches?",
        ["CISO", "CTO", "Security Operations Lead", "Any team lead"],
        2,
    ),
    # --- Section 7: Cloud Security ---
    make_q(
        "What is the internal codename for AWS?",
        ["Nimbus", "Cumulus", "Stratos", "Cirrus"],
        2,
    ),
    make_q(
        "What is the internal codename for Google Cloud Platform?",
        ["Stratos", "Nimbus", "Cumulus", "Alto"],
        2,
    ),
    make_q(
        "What is the internal codename for Microsoft Azure?",
        ["Stratos", "Cumulus", "Nimbus", "Cirrus"],
        2,
    ),
    make_q(
        "Through which gateway must all cloud access go?",
        ["Nexus VPN Gateway", "Nexus Cloud Gateway", "Nexus Access Proxy", "Nexus Secure Bridge"],
        1,
    ),
    make_q(
        "What Infrastructure as Code tool is mandated for production deployments?",
        ["CloudFormation", "Pulumi", "Terraform", "Ansible"],
        2,
    ),
    make_q(
        "How often must container images be scanned after initial deployment?",
        ["Daily", "Weekly", "Biweekly", "Monthly"],
        1,
    ),
    make_q(
        "What secrets management tool does Nexus use?",
        ["AWS Secrets Manager", "Nexus Vault", "CyberArk", "Azure Key Vault"],
        1,
    ),
    make_q(
        "What cost variance percentage triggers a cloud cost anomaly alert?",
        ["5%", "10%", "15%", "20%"],
        2,
    ),
    make_q(
        "Which regions are included in 'Region Alpha'?",
        ["US-East-1 and US-West-2", "US-East-1 and EU-West-1", "EU-West-1 and AP-Southeast-1", "US-East-1 only"],
        1,
    ),
    make_q(
        "Which additional region is included in 'Region Beta'?",
        ["US-West-2", "EU-Central-1", "AP-Southeast-1", "AP-Northeast-1"],
        2,
    ),
    make_q(
        "Is manual cloud console access permitted in production?",
        ["Yes, with MFA", "Yes, with approval", "No, prohibited", "Only for emergencies"],
        2,
    ),
    make_q(
        "How long are cloud access logs retained?",
        ["6 months", "1 year", "2 years", "5 years"],
        2,
    ),
    make_q(
        "What type of infrastructure is required for C3+ cloud workloads?",
        ["Containerized", "Serverless", "Immutable", "Multi-region"],
        2,
    ),
    make_q(
        "What resource tagging fields are mandatory for cloud resources?",
        ["Owner and cost-center only", "Owner, cost-center, classification level, and environment", "Owner and classification only", "Cost-center and environment only"],
        1,
    ),
    make_q(
        "Where can C2 data be stored geographically?",
        ["Region Alpha only", "Region Alpha and Region Beta", "Any approved region worldwide", "No restrictions"],
        1,
    ),
    # --- Section 8: Compliance & Audit ---
    make_q(
        "How often are internal security audits conducted?",
        ["Quarterly", "Every 6 months", "Annually", "Every 18 months"],
        1,
    ),
    make_q(
        "Which firm conducts the annual external audit?",
        ["Sentinel Cyber Assurance", "Meridian Assurance Partners", "Deloitte Cyber", "KPMG Security"],
        1,
    ),
    make_q(
        "How many surprise compliance checks must occur per year?",
        ["1", "2", "4", "6"],
        1,
    ),
    make_q(
        "Within how many days must critical audit findings be remediated?",
        ["7 days", "15 days", "30 days", "45 days"],
        1,
    ),
    make_q(
        "Within how many days must high audit findings be remediated?",
        ["15 days", "30 days", "45 days", "60 days"],
        1,
    ),
    make_q(
        "How long are security event logs retained?",
        ["6 months", "1 year", "2 years", "5 years"],
        2,
    ),
    make_q(
        "How long are financial system logs retained?",
        ["2 years", "5 years", "7 years", "10 years"],
        2,
    ),
    make_q(
        "How often are SIEM correlation rules reviewed?",
        ["Weekly", "Biweekly", "Monthly", "Quarterly"],
        2,
    ),
    make_q(
        "When must security awareness training be completed for new hires?",
        ["Before start date", "Within 7 days", "Within 30 days", "Within 60 days"],
        2,
    ),
    make_q(
        "How often are phishing simulation campaigns conducted?",
        ["Weekly", "Biweekly", "Monthly", "Quarterly"],
        2,
    ),
    make_q(
        "What is the acceptable phishing click rate threshold?",
        ["Below 2%", "Below 5%", "Below 8%", "Below 10%"],
        1,
    ),
    make_q(
        "What risk assessment methodology does Nexus use?",
        ["FAIR", "OCTAVE", "Nexus Risk Quantification Model (NRQM)", "ISO 27005"],
        2,
    ),
    make_q(
        "What is the NRQM risk appetite threshold?",
        ["Below 2.5", "Below 3.0", "Below 3.5", "Below 4.0"],
        2,
    ),
    make_q(
        "What is the NRQM scale range?",
        ["0-5", "0-10", "1-10", "1-100"],
        2,
    ),
    make_q(
        "What is the maximum validity of a policy exception?",
        ["3 months", "6 months", "12 months", "Indefinite with annual review"],
        1,
    ),
    make_q(
        "How often is the security policy itself reviewed?",
        ["Every 6 months", "Every 12 months", "Every 18 months", "Every 24 months"],
        1,
    ),
    make_q(
        "What happens on a third policy violation?",
        ["Mandatory training", "Written warning", "Disciplinary action", "Immediate termination"],
        2,
    ),
    make_q(
        "Within what timeframe must employees exceeding phishing click threshold complete remedial training?",
        ["24 hours", "3 days", "1 week", "2 weeks"],
        2,
    ),
    make_q(
        "How often must the security team complete certification training?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        1,
    ),
    make_q(
        "Within how many days must medium audit findings be remediated?",
        ["30 days", "45 days", "60 days", "90 days"],
        2,
    ),
    make_q(
        "What happens on a first policy violation?",
        ["Verbal warning", "Written warning", "Mandatory training", "Suspension"],
        1,
    ),
    make_q(
        "What happens on a second policy violation?",
        ["Written warning", "Mandatory training", "Suspension", "Termination"],
        1,
    ),
    # --- Section 9: Physical Security ---
    make_q(
        "What is Zone D designated for?",
        ["General office space", "Server racks and network equipment", "C4 data storage and key management hardware", "Reception and visitor lobby"],
        2,
    ),
    make_q(
        "What authentication is required for Zone C access?",
        ["Badge only", "Badge plus biometric", "Badge plus biometric plus escort", "Badge plus PIN"],
        1,
    ),
    make_q(
        "What authentication is required for Zone D access?",
        ["Badge plus biometric", "Badge plus biometric plus escort", "Badge plus biometric plus escort plus dual-person rule", "Biometric only"],
        2,
    ),
    make_q(
        "How long is CCTV footage retained for Zone C and Zone D?",
        ["30 days", "60 days", "90 days", "180 days"],
        2,
    ),
    make_q(
        "How long is CCTV footage retained for Zone A and Zone B?",
        ["7 days", "14 days", "30 days", "60 days"],
        2,
    ),
    make_q(
        "What temperature range must be maintained in server rooms?",
        ["15-20 degrees Celsius", "18-24 degrees Celsius", "20-26 degrees Celsius", "16-22 degrees Celsius"],
        1,
    ),
    make_q(
        "What humidity range must be maintained in server rooms?",
        ["30-50%", "35-55%", "40-60%", "45-65%"],
        2,
    ),
    make_q(
        "What fire suppression system is used in Zone C and Zone D?",
        ["Sprinkler system", "Halon 1301", "FM-200", "Novec 1230"],
        2,
    ),
    make_q(
        "What is the minimum UPS runtime for server rooms?",
        ["10 minutes", "15 minutes", "30 minutes", "60 minutes"],
        2,
    ),
    make_q(
        "Within how many seconds must the generator failover activate?",
        ["5 seconds", "10 seconds", "30 seconds", "60 seconds"],
        1,
    ),
    make_q(
        "What is the minimum data center tier required?",
        ["Tier I", "Tier II", "Tier III", "Tier IV"],
        2,
    ),
    make_q(
        "Where are visitor accesses logged?",
        ["Nexus Security Log", "Nexus Access Ledger", "Nexus Visitor Registry", "Security Operations Dashboard"],
        1,
    ),
    make_q(
        "Starting from which zone is visitor escort required?",
        ["Zone A", "Zone B", "Zone C", "Zone D"],
        1,
    ),
    make_q(
        "What is Zone A designated for?",
        ["General office space", "Reception and visitor lobby", "Server rooms", "Executive offices"],
        1,
    ),
    make_q(
        "What is Zone C designated for?",
        ["General office space", "Executive offices", "Server racks, network equipment, UPS systems", "C4 data vault"],
        2,
    ),
    make_q(
        "What types of biometric authentication are accepted for Zone C?",
        ["Fingerprint only", "Fingerprint or iris scan", "Iris scan only", "Fingerprint or facial recognition"],
        1,
    ),
    make_q(
        "What minimum seismic rating is required for data centers?",
        ["Zone 1", "Zone 2", "Zone 3", "Zone 4"],
        2,
    ),
    make_q(
        "What power redundancy is required for data centers?",
        ["Single feed with UPS", "Dual feeds from the same substation", "Dual feeds from independent substations", "Triple feeds"],
        2,
    ),
    # --- Section 10: Employee Security ---
    make_q(
        "How often must background checks be refreshed?",
        ["Every year", "Every 2 years", "Every 3 years", "Every 5 years"],
        2,
    ),
    make_q(
        "How many professional references are required during hiring?",
        ["1", "2", "3", "4"],
        1,
    ),
    make_q(
        "What office processes security clearances for C3+ access?",
        ["Nexus Security Office", "Nexus Clearance Office", "Nexus HR Security", "Nexus Access Bureau"],
        1,
    ),
    make_q(
        "In which type of workspace is the clean desk policy mandatory?",
        ["All workspaces", "C2 and above workspaces", "C3 and above workspaces", "Only Zone C and D"],
        1,
    ),
    make_q(
        "How many security champions are designated per team?",
        ["1", "2", "3", "Varies by team size"],
        0,
    ),
    make_q(
        "How often do security champions receive specialized training?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        1,
    ),
    make_q(
        "Within how many hours must all corporate devices be returned after departure?",
        ["4 hours", "12 hours", "24 hours", "48 hours"],
        2,
    ),
    make_q(
        "How long is the post-departure monitoring period for privileged account holders?",
        ["30 days", "60 days", "90 days", "180 days"],
        2,
    ),
    make_q(
        "Is printing C2+ documents at home permitted?",
        ["Yes, with encryption", "Yes, with manager approval", "No, prohibited", "Only for C2 level"],
        2,
    ),
    make_q(
        "What encryption standard is required for home wireless networks?",
        ["WPA2-Personal", "WPA2-Enterprise", "WPA3", "Any encryption"],
        2,
    ),
    make_q(
        "How often must home network router firmware be updated?",
        ["Monthly", "Quarterly", "Semi-annually", "Annually"],
        1,
    ),
    make_q(
        "Is remote work from public Wi-Fi permitted?",
        ["Yes, with VPN", "Yes, with MFA", "No, prohibited", "Only for C0 data"],
        2,
    ),
    make_q(
        "What must be signed before access to C2+ classified systems?",
        ["Non-disclosure agreement", "Acceptable use agreement", "Intellectual property agreement", "Background check consent"],
        2,
    ),
    make_q(
        "How often must the acceptable use agreement be signed?",
        ["Once at hiring", "Annually", "Every 2 years", "Every policy revision"],
        1,
    ),
    make_q(
        "Who is the CISO of Nexus Systems Corporation?",
        ["Dr. Marcus Chen", "Dr. Elara Vasquez", "Dr. Sarah Mitchell", "Dr. James Rodriguez"],
        1,
    ),
    make_q(
        "When did the current version of the NCF become effective?",
        ["January 1, 2024", "March 15, 2024", "July 1, 2024", "October 1, 2024"],
        1,
    ),
    make_q(
        "What is the classification level of the NCF document itself?",
        ["C0 (Public)", "C1 (Internal)", "C2 (Confidential)", "C3 (Restricted)"],
        1,
    ),
]


# === VALIDATION QUESTIONS ===
# Test different aspects/angles of the same policy.
# No direct overlap with training questions.

VAL_QUESTIONS = [
    # Cross-cutting / scenario-based questions
    make_q(
        "A new employee needs access to C3 systems. Which of these is NOT required?",
        ["Security clearance from Nexus Clearance Office", "Background check before start date", "CISO personal approval", "Intellectual property agreement"],
        2,
    ),
    make_q(
        "What is the total number of data classification levels in the NCF?",
        ["3", "4", "5", "6"],
        2,
    ),
    make_q(
        "If an administrator's account is locked out, what is the total wait time before they can try again?",
        ["30 minutes", "1 hour", "2 hours", "4 hours"],
        3,
    ),
    make_q(
        "Which of the following is the LONGEST certificate lifetime allowed?",
        ["Server certificate (398 days)", "Client certificate (365 days)", "Code signing certificate (730 days)", "All have the same lifetime"],
        2,
    ),
    make_q(
        "A SEV-2 incident occurs at 2:00 PM. By what time must the initial response occur?",
        ["2:15 PM", "2:30 PM", "3:00 PM", "4:00 PM"],
        2,
    ),
    make_q(
        "Which network segment allows restricted internet access via proxy?",
        ["RED", "AMBER", "BLUE", "GREEN"],
        1,
    ),
    make_q(
        "What is the combined number of failed login attempts before lockout for admin + standard user?",
        ["6", "7", "8", "10"],
        2,
    ),
    make_q(
        "How many approved cloud providers does Nexus allow?",
        ["1", "2", "3", "4"],
        2,
    ),
    make_q(
        "Which of these is a valid Zone D access requirement?",
        ["Badge and PIN", "Badge, biometric, and manager approval", "Badge, biometric, security escort, and dual-person rule", "Badge, biometric, and CISO approval"],
        2,
    ),
    make_q(
        "A contractor's password must be at least how many characters longer than a standard user's?",
        ["2 characters longer", "Same length", "2 characters shorter", "4 characters shorter"],
        0,
    ),
    make_q(
        "What is the difference in retention periods between C1 and C2 data?",
        ["1 year", "2 years", "3 years", "5 years"],
        1,
    ),
    make_q(
        "Which segments require host-based IDS according to the NCF?",
        ["All segments", "AMBER and RED only", "C3+ systems (AMBER and RED)", "RED only"],
        2,
    ),
    make_q(
        "If a critical vulnerability is found on Monday at 9 AM, by when must it be patched?",
        ["Monday 5 PM", "Tuesday 9 AM", "Wednesday 9 AM", "Friday 9 AM"],
        1,
    ),
    make_q(
        "What Infrastructure as Code tools are allowed for Nexus cloud deployments?",
        ["Terraform and Pulumi", "Terraform only", "CloudFormation and Terraform", "Any IaC tool"],
        1,
    ),
    make_q(
        "How many physical security zones does the NCF define?",
        ["2", "3", "4", "5"],
        2,
    ),
    make_q(
        "Which of the following combinations is correct for the VPN?",
        ["OpenVPN with 128-bit keys, re-key every 60 min", "WireGuard with 256-bit keys, re-key every 90 min", "IPSec with 256-bit keys, re-key every 120 min", "WireGuard with 128-bit keys, re-key every 60 min"],
        1,
    ),
    make_q(
        "An employee is terminated at 10:00 AM. By when must their access be revoked?",
        ["10:30 AM", "12:00 PM", "2:00 PM", "6:00 PM"],
        2,
    ),
    make_q(
        "What is the frequency difference between RED and AMBER segment penetration tests?",
        ["RED is weekly, AMBER is monthly", "RED is monthly, AMBER is quarterly", "RED is quarterly, AMBER is annually", "Both are monthly"],
        1,
    ),
    make_q(
        "Which type of data can NEVER leave Region Alpha?",
        ["C2", "C3", "C4", "All classified data"],
        2,
    ),
    make_q(
        "What is the minimum number of Argon2id iterations required by the NCF?",
        ["1", "2", "3", "5"],
        2,
    ),
    make_q(
        "Which of the following is NOT an approved MFA method?",
        ["YubiKey 5 series", "TOTP authenticator app", "SMS verification", "All listed are approved"],
        2,
    ),
    make_q(
        "What is the total number of incident severity levels?",
        ["3", "4", "5", "6"],
        1,
    ),
    make_q(
        "A post-incident review for a SEV-3 event must happen within what timeframe?",
        ["24 hours", "72 hours", "1 week", "2 weeks"],
        3,
    ),
    make_q(
        "What distinguishes C4 data destruction from C3 data destruction?",
        ["C4 requires cryptographic erasure, C3 requires standard deletion", "Both require physical destruction", "C4 requires physical destruction, C3 requires cryptographic erasure", "No difference"],
        2,
    ),
    make_q(
        "How many concurrent VPN sessions can a single user maintain?",
        ["1", "2", "3", "Unlimited"],
        1,
    ),
    make_q(
        "What version identifier is the current NCF?",
        ["NCF-2023-R2", "NCF-2024-R3", "NCF-2024-R1", "NCF-2025-R1"],
        1,
    ),
    make_q(
        "For how long after departure are privileged users' accounts monitored?",
        ["30 days", "60 days", "90 days", "No monitoring"],
        2,
    ),
    make_q(
        "Which wireless SSID should corporate employees connect to?",
        ["NexusGuest", "NexusSecure", "NexusCorp", "NexusEmployee"],
        1,
    ),
    make_q(
        "What is the maximum retention period difference between security logs and financial logs?",
        ["3 years", "5 years", "7 years", "They are the same"],
        1,
    ),
    make_q(
        "A risk score of 4.0 on the NRQM scale would be classified as:",
        ["Acceptable", "Above risk appetite (unacceptable)", "Negligible", "Moderate but acceptable"],
        1,
    ),
    make_q(
        "Which of the following pairs share the same December 2025 phase-out date?",
        ["SHA-256 and RSA-4096", "TLS 1.2 and RSA-4096", "CBC mode and TLS 1.2", "SHA-256 and TLS 1.2"],
        1,
    ),
    make_q(
        "A standard user account with 4 failed login attempts would:",
        ["Be locked out for 30 minutes", "Receive a warning but not be locked", "Be locked out for 4 hours", "Have the SOC alerted"],
        1,
    ),
    make_q(
        "Which of the following cloud operations requires CISO approval?",
        ["Deploying to Region Alpha", "Multi-region replication of C3 data", "Using Terraform for IaC", "Container image scanning"],
        1,
    ),
    make_q(
        "The NCF names two specific external firms. Which are they?",
        ["Sentinel Cyber and Meridian Assurance", "CrowdStrike and Mandiant", "Deloitte and KPMG", "PwC and EY"],
        0,
    ),
    make_q(
        "What is the minimum duration between full internal security audits?",
        ["3 months", "6 months", "9 months", "12 months"],
        1,
    ),
    make_q(
        "Which security control applies to both Zone C and Zone D equally?",
        ["Dual-person access rule", "Security escort requirement", "FM-200 fire suppression", "Badge-only access"],
        2,
    ),
    make_q(
        "What is the relationship between Nexus Vault and HashiCorp Vault?",
        ["They are the same product", "Nexus Vault is a derivative of HashiCorp Vault", "HashiCorp Vault is a derivative of Nexus Vault", "They are competitors"],
        1,
    ),
    make_q(
        "Which of the following is NOT a mandatory cloud resource tag?",
        ["Owner", "Department", "Classification level", "Environment"],
        1,
    ),
    make_q(
        "The NCF requires how many surprise compliance checks per year?",
        ["At least 1", "At least 2", "At least 4", "Monthly"],
        1,
    ),
    make_q(
        "An employee who clicks a phishing link in a simulation must complete remedial training within:",
        ["24 hours", "3 business days", "1 week", "2 weeks"],
        2,
    ),
    make_q(
        "Which hash algorithm is specifically prohibited for new implementations?",
        ["SHA-384", "SHA-512", "SHA-256", "BLAKE3"],
        2,
    ),
    make_q(
        "What type of infrastructure is mandated for all production cloud deployments?",
        ["Containerized", "Serverless", "Infrastructure as Code (Terraform)", "Microservices"],
        2,
    ),
    make_q(
        "Which classification levels have no geographic data restrictions?",
        ["C0 only", "C0 and C1", "C0, C1, and C2", "All levels in approved regions"],
        1,
    ),
    make_q(
        "What is the minimum data center redundancy standard?",
        ["N (no redundancy)", "N+1 (Tier III)", "2N (Tier IV)", "2N+1"],
        1,
    ),
    make_q(
        "The Nexus EDR platform is described as a derivative of which commercial product?",
        ["Microsoft Defender", "SentinelOne", "CrowdStrike", "Carbon Black"],
        2,
    ),
    make_q(
        "How soon after hiring must new employees complete security awareness training?",
        ["Before start date", "Within 7 days", "Within 30 days", "Within 90 days"],
        2,
    ),
    make_q(
        "What is the maximum number of hosts per VLAN in any segment?",
        ["100", "200", "250", "500"],
        2,
    ),
    make_q(
        "The NCF effective date is:",
        ["January 1, 2024", "March 15, 2024", "June 1, 2024", "September 1, 2024"],
        1,
    ),
    make_q(
        "Which encryption algorithm must C4 data use for both at-rest AND in-transit protection?",
        ["AES-128-GCM", "AES-256-GCM", "ChaCha20-Poly1305", "RSA-4096"],
        1,
    ),
    make_q(
        "What log integrity mechanism is required?",
        ["Digital signatures", "Cryptographic hash chain verification", "Immutable storage", "Write-once media"],
        1,
    ),
]


def main():
    rng = random.Random(42)

    # Finalize questions: shuffle option order so correct answer is uniformly distributed
    train = [finalize_q(q, rng) for q in TRAIN_QUESTIONS]
    rng.shuffle(train)
    val = [finalize_q(q, rng) for q in VAL_QUESTIONS]

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "train.json").write_text(json.dumps(train, indent=2))
    (out_dir / "val.json").write_text(json.dumps(val, indent=2))

    print(f"Generated {len(train)} training questions -> {out_dir / 'train.json'}")
    print(f"Generated {len(val)} validation questions -> {out_dir / 'val.json'}")

    # Verify no duplicate prompts within each set
    train_prompts = set(q["prompt"] for q in train)
    val_prompts = set(q["prompt"] for q in val)
    assert len(train_prompts) == len(train), "Duplicate training prompts found!"
    assert len(val_prompts) == len(val), "Duplicate validation prompts found!"
    print("No duplicate prompts found.")

    # Verify answer distribution is roughly uniform
    from collections import Counter
    train_dist = Counter(q["answer"] for q in train)
    val_dist = Counter(q["answer"] for q in val)
    print(f"Train answer distribution: {dict(sorted(train_dist.items()))}")
    print(f"Val answer distribution:   {dict(sorted(val_dist.items()))}")


if __name__ == "__main__":
    main()
