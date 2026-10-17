from pathlib import Path
import tempfile
import zipfile

import tellurium as te


BASE_DIR = Path(__file__).resolve().parents[2]

OMEX_PATH = (
    BASE_DIR
    / "models"
    / "HPO"
    / "MODEL1311220000.3.omex"
)

SBML_FILENAME = "BIOMD0000000494_url.xml"


class RoblitzModelRunner:
    """
    Loads and runs the original Röblitz 2013
    menstrual-cycle / HPO model from the OMEX package.
    """

    def __init__(self, omex_path=OMEX_PATH):

        self.omex_path = Path(omex_path)

        if not self.omex_path.exists():
            raise FileNotFoundError(
                f"OMEX file not found: {self.omex_path}"
            )

        self.model = self._load_model()

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------

    def _load_model(self):

        with zipfile.ZipFile(
            self.omex_path,
            "r",
        ) as archive:

            if SBML_FILENAME not in archive.namelist():
                raise FileNotFoundError(
                    f"{SBML_FILENAME} not found inside OMEX."
                )

            sbml_data = archive.read(
                SBML_FILENAME
            )

        with tempfile.NamedTemporaryFile(
            suffix=".xml",
            delete=False,
        ) as temp:

            temp.write(sbml_data)
            sbml_path = temp.name

        try:
            model = te.loadSBMLModel(
                sbml_path
            )
        finally:
            Path(sbml_path).unlink(
                missing_ok=True
            )

        # Provisional numerical mitigation for the scheduled
        # Ago administration event near time 91.
        # Tested at 1e-10: CVODE warnings disappeared and the
        # tested LH trajectory remained nearly unchanged.
        model.integrator.absolute_tolerance = 1e-10

        return model

    # ---------------------------------------------------------
    # MODEL INFORMATION
    # ---------------------------------------------------------

    def get_species(self):

        return [
            variable.name
            for variable
            in self.model.getFloatingSpeciesConcentrations()
        ]

    def get_species_ids(self):

        return list(
            self.model.getFloatingSpeciesIds()
        )

    def get_parameters(self):

        return list(
            self.model.getGlobalParameterIds()
        )

    # ---------------------------------------------------------
    # FRESH MODEL
    # ---------------------------------------------------------

    def create_fresh_model(self):
        """
        Create a completely fresh instance of the Röblitz model.

        This prevents parameter changes made for one scenario
        from affecting another scenario.
        """

        return self._load_model()

    # ---------------------------------------------------------
    # INITIAL STATE
    # ---------------------------------------------------------

    def get_initial_state(self):

        return self.model.getFloatingSpeciesConcentrations()

    # ---------------------------------------------------------
    # STATE SAVE / LOAD
    # ---------------------------------------------------------

    def save_state(self, filename):

        self.model.saveState(
            str(filename)
        )

    def load_state(self, filename):

        self.model.loadState(
            str(filename)
        )

    # ---------------------------------------------------------
    # SIMULATION
    # ---------------------------------------------------------

    def simulate(
        self,
        start=0,
        end=120,
        points=1201,
    ):

        result = self.model.simulate(
            start,
            end,
            points,
        )

        return result