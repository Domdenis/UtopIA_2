"""
UtopIA — PatientState
Structure de données partagée entre toutes les étapes du workflow.
Stockée dans st.session_state["patient"].
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class PatientState:
    # ── Identification ────────────────────────────────────────────────
    nom: str = ""
    prenom: str = ""
    date_naissance: str = ""
    adresse: str = ""

    # ── Facteurs personnels (OTIPM Phase 1) ──────────────────────────
    age: Optional[int] = None
    sexe: str = ""
    diagnostic: str = ""
    lateralite: str = ""
    capacites_physiques: str = ""
    capacites_cognitives: bool = True
    comorbidites: str = ""
    poids: Optional[float] = None
    taille: Optional[float] = None

    # Mesures anthropométriques
    largeur_bassin: Optional[float] = None
    longueur_cuisses: Optional[float] = None
    longueur_creux_poplite_pied: Optional[float] = None
    hauteur_coude: Optional[float] = None
    hauteur_omoplate: Optional[float] = None
    largeur_tronc: Optional[float] = None
    epaisseur_tronc: Optional[float] = None
    hauteur_epaule: Optional[float] = None

    # ── Usage et activités ────────────────────────────────────────────
    premiere_acquisition: bool = True
    ancien_vph_categorie: str = ""
    activites: list = field(default_factory=list)  # vie_quotidienne, travail, loisirs
    temps_utilisation: str = ""
    deplacements: list = field(default_factory=list)  # intérieur, extérieur, tout-terrain

    # ── Facteurs environnementaux ─────────────────────────────────────
    lieu_vie: str = ""
    description_logement: str = ""
    escaliers: bool = False
    ascenseur: bool = False
    terrain_accidente: bool = False
    environnement_humain: str = ""
    activite_professionnelle: bool = False
    conduite_vehicule: bool = False
    transports_commun: bool = False

    # ── Situation de santé ────────────────────────────────────────────
    situation_sante: str = ""
    caractere_evolutif: str = ""

    # ── Synthèse et résultats ─────────────────────────────────────────
    synthese_demande: str = ""
    modele_conceptuel_choisi: str = ""
    justification_modele: str = ""
    diagnostic_ergo: str = ""

    # ── Préconisations AT ─────────────────────────────────────────────
    categorie_vph_recommandee: str = ""
    propositions_at: list = field(default_factory=list)
    mode_prise_en_charge: str = ""  # achat / LCD / LLD

    # ── Post-essais ───────────────────────────────────────────────────
    at_essayees: list = field(default_factory=list)
    at_retenue: str = ""
    observations_essais: str = ""
    motifs_rejet: str = ""
    reglages_definitifs: str = ""

    # ── Dossier final ─────────────────────────────────────────────────
    argumentaire_cpam: str = ""
    dossier_complet: str = ""

    # ── Méta ──────────────────────────────────────────────────────────
    date_creation: str = field(default_factory=lambda: datetime.now().strftime("%d/%m/%Y"))
    statut: str = "recueil"  # recueil → preconisation → essais → dossier

    def to_context_summary(self) -> str:
        """Résumé du profil pour injection dans les prompts LLM."""
        parts = []
        if self.nom or self.prenom:
            parts.append(f"Patient : {self.prenom} {self.nom}")
        if self.age:
            parts.append(f"Âge : {self.age} ans | Sexe : {self.sexe}")
        if self.diagnostic:
            parts.append(f"Diagnostic : {self.diagnostic}")
        if self.situation_sante:
            parts.append(f"Situation de santé : {self.situation_sante}")
        if self.capacites_physiques:
            parts.append(f"Capacités physiques : {self.capacites_physiques}")
        if self.lieu_vie:
            parts.append(f"Lieu de vie : {self.lieu_vie}")
        if self.activites:
            parts.append(f"Activités : {', '.join(self.activites)}")
        if self.deplacements:
            parts.append(f"Déplacements : {', '.join(self.deplacements)}")
        if self.synthese_demande:
            parts.append(f"Synthèse : {self.synthese_demande}")
        mesures = []
        if self.largeur_bassin:
            mesures.append(f"bassin {self.largeur_bassin}cm")
        if self.poids:
            mesures.append(f"poids {self.poids}kg")
        if mesures:
            parts.append(f"Mesures : {', '.join(mesures)}")
        return "\n".join(parts)
