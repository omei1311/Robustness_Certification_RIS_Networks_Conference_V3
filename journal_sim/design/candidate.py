from dataclasses import dataclass, field
from journal_sim.core.models import Configuration


@dataclass
class Candidate:
    index: int
    seed: int
    configuration: Configuration
    H_hat: object
    rate: float
    system_power: float
    wee: float
    sinr_min: float
    nominal_feasible: bool
    certificate: object
    epsilon_est: float = 0.
    metadata: dict = field(default_factory=dict)

    @property
    def epsilon_cert(self):
        return self.certificate.epsilon_cert

    @property
    def valid_certificate(self):
        return (self.nominal_feasible and self.certificate.reusable and
                self.certificate.matches(self.configuration.w, self.H_hat, self.configuration.theta, self.metadata["cfg"]))

    def to_record(self):
        return dict(candidate_index=self.index, candidate_seed=self.seed,
                    configuration_id=self.configuration.configuration_id,
                    rate=self.rate, system_power=self.system_power, wee=self.wee,
                    spectral_wee=self.wee / self.metadata["cfg"].bandwidth_hz,
                    sinr_min=self.sinr_min, nominal_feasible=self.nominal_feasible,
                    epsilon_est=self.epsilon_est,
                    **{k: v for k, v in self.metadata.items() if k != "cfg"},
                    **self.certificate.to_dict())
