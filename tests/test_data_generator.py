"""
Unit tests for the data generator module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_generator import HealthcareDataGenerator
from src.config import DATA_CONFIG

class TestHealthcareDataGenerator:
    """Test cases for HealthcareDataGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DATA_CONFIG.copy()
        self.config["n_samples"] = 1000  # Use smaller sample for tests
        self.generator = HealthcareDataGenerator(self.config)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.config == self.config
        assert self.generator.random_state is not None
    
    def test_generate_demographics(self):
        """Test demographic generation."""
        demographics = self.generator.generate_demographics(100)
        
        assert isinstance(demographics, pd.DataFrame)
        assert len(demographics) == 100
        assert all(col in demographics.columns for col in ['age', 'gender', 'bmi'])
        
        # Check value ranges
        assert demographics['age'].min() >= 18
        assert demographics['age'].max() <= 85
        assert demographics['bmi'].min() >= 18
        assert demographics['bmi'].max() <= 45
        assert set(demographics['gender'].unique()).issubset({0, 1})
    
    def test_generate_medical_conditions(self):
        """Test medical conditions generation."""
        demographics = self.generator.generate_demographics(100)
        medical_conditions = self.generator.generate_medical_conditions(demographics)
        
        assert isinstance(medical_conditions, pd.DataFrame)
        assert len(medical_conditions) == 100
        assert all(col in medical_conditions.columns for col in ['smoker', 'diabetes', 'hypertension'])
        
        # Check binary values
        for col in ['smoker', 'diabetes', 'hypertension']:
            assert set(medical_conditions[col].unique()).issubset({0, 1})
    
    def test_generate_treatment_assignment(self):
        """Test treatment assignment generation."""
        demographics = self.generator.generate_demographics(100)
        medical_conditions = self.generator.generate_medical_conditions(demographics)
        treatment = self.generator.generate_treatment_assignment(demographics, medical_conditions)
        
        assert isinstance(treatment, pd.Series)
        assert len(treatment) == 100
        assert treatment.name == 'treatment'
        assert set(treatment.unique()).issubset({0, 1})
    
    def test_generate_outcome(self):
        """Test outcome generation."""
        demographics = self.generator.generate_demographics(100)
        medical_conditions = self.generator.generate_medical_conditions(demographics)
        treatment = self.generator.generate_treatment_assignment(demographics, medical_conditions)
        outcome = self.generator.generate_outcome(demographics, medical_conditions, treatment)
        
        assert isinstance(outcome, pd.Series)
        assert len(outcome) == 100
        assert outcome.name == 'outcome'
        assert set(outcome.unique()).issubset({0, 1})
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        data = self.generator.generate_dataset(100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        expected_columns = ['age', 'gender', 'bmi', 'smoker', 'diabetes', 
                           'hypertension', 'treatment', 'outcome']
        assert all(col in data.columns for col in expected_columns)
        
        # Check for derived columns
        assert 'age_group' in data.columns
        assert 'bmi_category' in data.columns
    
    def test_get_true_treatment_effect(self):
        """Test getting true treatment effect."""
        true_effect = self.generator.get_true_treatment_effect()
        assert isinstance(true_effect, float)
        assert true_effect == self.config["treatment_effect"]
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        data = self.generator.generate_dataset(100)
        summary = self.generator.get_data_summary(data)
        
        assert isinstance(summary, dict)
        assert 'n_samples' in summary
        assert 'treatment_rate' in summary
        assert 'outcome_rate' in summary
        assert summary['n_samples'] == 100
    
    def test_data_consistency(self):
        """Test data consistency across generation."""
        data1 = self.generator.generate_dataset(100)
        data2 = self.generator.generate_dataset(100)
        
        # With same random state, should be identical
        assert data1.equals(data2)
    
    def test_different_sample_sizes(self):
        """Test generation with different sample sizes."""
        sizes = [10, 100, 1000]
        for size in sizes:
            data = self.generator.generate_dataset(size)
            assert len(data) == size
    
    def test_config_modification(self):
        """Test generator with modified config."""
        modified_config = self.config.copy()
        modified_config["treatment_effect"] = 0.25
        modified_generator = HealthcareDataGenerator(modified_config)
        
        assert modified_generator.get_true_treatment_effect() == 0.25
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            self.generator.generate_demographics(-1)
        
        with pytest.raises(ValueError):
            self.generator.generate_demographics(0) 