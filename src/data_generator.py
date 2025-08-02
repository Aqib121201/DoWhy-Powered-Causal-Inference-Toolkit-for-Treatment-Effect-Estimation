"""
Synthetic data generator for healthcare intervention studies.

This module generates realistic synthetic data for causal inference studies,
simulating a healthcare intervention scenario with proper causal relationships
and confounding variables.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from .config import DATA_CONFIG

logger = logging.getLogger(__name__)

class HealthcareDataGenerator:
    """
    Generates synthetic healthcare data for causal inference studies.
    
    This class creates realistic patient data with proper causal relationships
    between demographic variables, medical conditions, treatment assignment,
    and health outcomes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration dictionary for data generation parameters
        """
        self.config = config or DATA_CONFIG
        self.random_state = np.random.RandomState(self.config["random_state"])
        
    def generate_demographics(self, n_samples: int) -> pd.DataFrame:
        """
        Generate demographic variables.
        
        Args:
            n_samples: Number of patients to generate
            
        Returns:
            DataFrame with demographic variables
        """
        logger.info(f"Generating demographics for {n_samples} patients")
        
        # Age: Normal distribution with realistic range
        age = self.random_state.normal(55, 15, n_samples)
        age = np.clip(age, 18, 85)
        
        # Gender: Binary with slight imbalance
        gender = self.random_state.binomial(1, 0.52, n_samples)
        
        # BMI: Normal distribution with realistic range
        bmi = self.random_state.normal(27, 5, n_samples)
        bmi = np.clip(bmi, 18, 45)
        
        demographics = pd.DataFrame({
            'age': age,
            'gender': gender,
            'bmi': bmi
        })
        
        return demographics
    
    def generate_medical_conditions(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """
        Generate medical conditions based on demographics.
        
        Args:
            demographics: DataFrame with age, gender, bmi
            
        Returns:
            DataFrame with medical conditions
        """
        logger.info("Generating medical conditions")
        
        n_samples = len(demographics)
        
        # Smoking: influenced by age and gender
        age_normalized = (demographics['age'] - 18) / (85 - 18)
        gender_effect = demographics['gender'] * 0.1
        smoking_prob = 0.3 - 0.2 * age_normalized + gender_effect
        smoking_prob = np.clip(smoking_prob, 0.05, 0.6)
        smoker = self.random_state.binomial(1, smoking_prob, n_samples)
        
        # Diabetes: influenced by age, BMI, and smoking
        bmi_normalized = (demographics['bmi'] - 18) / (45 - 18)
        diabetes_prob = 0.1 + 0.3 * age_normalized + 0.2 * bmi_normalized + 0.1 * smoker
        diabetes_prob = np.clip(diabetes_prob, 0.02, 0.4)
        diabetes = self.random_state.binomial(1, diabetes_prob, n_samples)
        
        # Hypertension: influenced by age, BMI, diabetes, and smoking
        hypertension_prob = (0.15 + 0.25 * age_normalized + 0.15 * bmi_normalized + 
                           0.2 * diabetes + 0.1 * smoker)
        hypertension_prob = np.clip(hypertension_prob, 0.05, 0.5)
        hypertension = self.random_state.binomial(1, hypertension_prob, n_samples)
        
        medical_conditions = pd.DataFrame({
            'smoker': smoker,
            'diabetes': diabetes,
            'hypertension': hypertension
        })
        
        return medical_conditions
    
    def generate_treatment_assignment(self, demographics: pd.DataFrame, 
                                    medical_conditions: pd.DataFrame) -> pd.Series:
        """
        Generate treatment assignment based on patient characteristics.
        
        Args:
            demographics: DataFrame with demographic variables
            medical_conditions: DataFrame with medical conditions
            
        Returns:
            Series with treatment assignment (0: control, 1: treatment)
        """
        logger.info("Generating treatment assignment")
        
        n_samples = len(demographics)
        
        # Treatment assignment influenced by multiple factors
        age_effect = (demographics['age'] - 50) / 35  # Centered and scaled
        bmi_effect = (demographics['bmi'] - 27) / 13.5  # Centered and scaled
        gender_effect = demographics['gender'] * 0.1
        diabetes_effect = medical_conditions['diabetes'] * 0.3
        hypertension_effect = medical_conditions['hypertension'] * 0.2
        smoker_effect = medical_conditions['smoker'] * 0.1
        
        # Combine effects with some randomness
        treatment_prob = (0.5 + 0.1 * age_effect + 0.05 * bmi_effect + 
                         gender_effect + diabetes_effect + hypertension_effect + 
                         smoker_effect + self.random_state.normal(0, 0.1, n_samples))
        
        treatment_prob = 1 / (1 + np.exp(-treatment_prob))  # Sigmoid function
        treatment = self.random_state.binomial(1, treatment_prob, n_samples)
        
        return pd.Series(treatment, name='treatment')
    
    def generate_outcome(self, demographics: pd.DataFrame, 
                        medical_conditions: pd.DataFrame,
                        treatment: pd.Series) -> pd.Series:
        """
        Generate health outcome based on all variables and treatment.
        
        Args:
            demographics: DataFrame with demographic variables
            medical_conditions: DataFrame with medical conditions
            treatment: Series with treatment assignment
            
        Returns:
            Series with health outcome (0: failure, 1: success)
        """
        logger.info("Generating health outcomes")
        
        n_samples = len(demographics)
        
        # Baseline outcome probability
        base_prob = 0.4
        
        # Effects of demographic variables
        age_effect = (demographics['age'] - 50) / 35 * 0.1
        bmi_effect = (demographics['bmi'] - 27) / 13.5 * 0.15
        gender_effect = demographics['gender'] * 0.05
        
        # Effects of medical conditions
        diabetes_effect = medical_conditions['diabetes'] * -0.2
        hypertension_effect = medical_conditions['hypertension'] * -0.15
        smoker_effect = medical_conditions['smoker'] * -0.1
        
        # Treatment effect (the causal effect we want to estimate)
        treatment_effect = treatment * self.config["treatment_effect"]
        
        # Confounding effects (variables that affect both treatment and outcome)
        confounding_effect = (self.config["confounding_strength"] * 
                             (age_effect + bmi_effect + diabetes_effect))
        
        # Combine all effects
        outcome_prob = (base_prob + age_effect + bmi_effect + gender_effect +
                       diabetes_effect + hypertension_effect + smoker_effect +
                       treatment_effect + confounding_effect)
        
        # Add noise
        noise = self.random_state.normal(0, self.config["noise_level"], n_samples)
        outcome_prob += noise
        
        # Ensure probabilities are in valid range
        outcome_prob = np.clip(outcome_prob, 0.01, 0.99)
        
        # Generate binary outcomes
        outcome = self.random_state.binomial(1, outcome_prob, n_samples)
        
        return pd.Series(outcome, name='outcome')
    
    def generate_dataset(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Generate complete healthcare dataset.
        
        Args:
            n_samples: Number of patients to generate (uses config default if None)
            
        Returns:
            Complete DataFrame with all variables
        """
        n_samples = n_samples or self.config["n_samples"]
        logger.info(f"Generating complete dataset with {n_samples} patients")
        
        # Generate all components
        demographics = self.generate_demographics(n_samples)
        medical_conditions = self.generate_medical_conditions(demographics)
        treatment = self.generate_treatment_assignment(demographics, medical_conditions)
        outcome = self.generate_outcome(demographics, medical_conditions, treatment)
        
        # Combine all data
        dataset = pd.concat([demographics, medical_conditions, treatment, outcome], axis=1)
        
        # Add some derived variables
        dataset['age_group'] = pd.cut(dataset['age'], 
                                     bins=[0, 35, 50, 65, 100], 
                                     labels=['18-35', '36-50', '51-65', '65+'])
        dataset['bmi_category'] = pd.cut(dataset['bmi'], 
                                        bins=[0, 18.5, 25, 30, 100], 
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        logger.info(f"Generated dataset with shape: {dataset.shape}")
        logger.info(f"Treatment rate: {dataset['treatment'].mean():.3f}")
        logger.info(f"Outcome rate: {dataset['outcome'].mean():.3f}")
        
        return dataset
    
    def get_true_treatment_effect(self) -> float:
        """
        Get the true treatment effect used in data generation.
        
        Returns:
            True average treatment effect
        """
        return self.config["treatment_effect"]
    
    def get_data_summary(self, dataset: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataset.
        
        Args:
            dataset: Generated dataset
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "n_samples": len(dataset),
            "n_features": len(dataset.columns),
            "treatment_rate": dataset['treatment'].mean(),
            "outcome_rate": dataset['outcome'].mean(),
            "age_mean": dataset['age'].mean(),
            "age_std": dataset['age'].std(),
            "bmi_mean": dataset['bmi'].mean(),
            "bmi_std": dataset['bmi'].std(),
            "smoker_rate": dataset['smoker'].mean(),
            "diabetes_rate": dataset['diabetes'].mean(),
            "hypertension_rate": dataset['hypertension'].mean(),
            "gender_distribution": dataset['gender'].value_counts().to_dict()
        }
        
        return summary 