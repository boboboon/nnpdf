"""
Module containing the classes related to the stopping alogirthm

In this module there are four Classes:

- FitState: this class contains the information of the fit
        for a given point in history
- FitHistory: this class contains the information necessary
        in order to reset the state of the fit to the point
        in which the history was saved.
        i.e., a list of FitStates
- Stopping: this class monitors the chi2 of the validation
        and training sets and decides when to stop
- Positivity: Decides whether a given point fullfills the positivity conditions
- Validation: Controls the NNPDF cross-validation algorithm

Note:
    There are situations in which the validation set is empty, in those cases
the training set is used as validation set.
This implies several changes in the behaviour of this class as the training chi2 will
now be monitored for stability.
    In order to parse the set of loss functions coming from the backend::MetaModel,
the function `parse_losses` relies on the fact that they are all suffixed with `_loss`
the validation case, instead, is suffixed with `val_loss`. In the particular casse in
which both training and validation model correspond to the same backend::MetaModel only
the `_loss` suffix can be found. This is taken into account by the class `Stopping`
which will tell `Validation` that no validation set was found and that the training is to
be used instead.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# Put a very big number here so that we for sure discard this run
# AND we have a clear marker that something went wrong, not just a bad fit
TERRIBLE_CHI2 = 1e10
INITIAL_CHI2 = 1e9

# Pass/veto keys
POS_OK = "POS_PASS"
POS_BAD = "POS_VETO"
THRESHOLD_POS = 1e-6


def parse_ndata(all_data):
    """
    Parses the list of dictionaries received from ModelTrainer into dictionaries
    containing only the name of the experiments and the number of points per replica

    Returns
    -------
        `tr_ndata`
            dictionary of {'exp' : np.ndarray}
        `vl_ndata`
            dictionary of {'exp' : np.ndarray}
        `pos_set`: list of the names of the positivity sets

    Note: if there is no validation (total number of val points == 0)
    then vl_ndata will point to tr_ndata
    """
    tr_ndata_dict = {}
    vl_ndata_dict = {}
    pos_set = []
    for dictionary in all_data:
        exp_name = dictionary["name"]
        if dictionary.get("count_chi2"):
            tr_ndata = dictionary["ndata"]
            vl_ndata = dictionary["ndata_vl"]
            if sum(tr_ndata) != 0:
                tr_ndata_dict[exp_name] = np.array(tr_ndata)
            if sum(vl_ndata) != 0:
                vl_ndata_dict[exp_name] = np.array(vl_ndata)
        if dictionary.get("positivity") and not dictionary.get("integrability"):
            pos_set.append(exp_name)
    if not vl_ndata_dict:
        vl_ndata_dict = None
    return tr_ndata_dict, vl_ndata_dict, pos_set


def parse_losses(history_object, data, suffix="loss"):
    """
    Receives an object containing the chi2
    Usually a history object, but it can come in the form of a dictionary.

    It loops over the dictionary and uses the npoints_data dictionary to
    normalize the chi2 and return backs a tuple (`total`, `tr_chi2`)

    Parameters
    ----------
        history_object: dict
            A history object dictionary
        data: dict
            dictionary with the name of the experiments to be taken into account
            and the number of datapoints of the experiments
        suffix: str (default: ``loss``)
            suffix of the loss layer, Keras default is _loss

    Returns
    -------
        total_loss: float
            Total value for the loss
        dict_chi2: dict
            dictionary of {'expname' : loss }
    """
    try:
        hobj = history_object.history
    except AttributeError:  # So it works whether we pass the out or the out.history
        hobj = history_object

    dict_chi2 = {}
    total_points = 0
    total_loss = np.zeros_like(hobj["loss"])
    for exp_name, npoints in data.items():
        loss = np.array(hobj[exp_name + f"_{suffix}"])
        dict_chi2[exp_name] = loss / np.maximum(npoints, 1)
        total_points += npoints
        total_loss += loss

    # By taking the loss from the history object we would be saving the total loss
    # including positivity sets and (if added/enabled) regularizsers
    # instead we want to restrict ourselves to the loss coming from experiments
    total_loss /= np.maximum(total_points, 1)
    dict_chi2["total"] = total_loss
    return total_loss, dict_chi2


class FitState:
    """
    Holds the state of the chi2 during the fit, for all replicas and one epoch

    Note: the training chi2 is computed before the update of the weights
    so it is the chi2 that informed the updated corresponding to this state.
    The validation chi2 instead is computed after the update of the weights.

    Parameters
    ----------
        training_info: dict
            all losses for the training model
        validation_info: dict
            all losses for the validation model
        training_loss: float
            total training loss, this can be given if per-exp``training_info``
            is not available
    """

    vl_ndata = None
    tr_ndata = None
    vl_suffix = None

    def __init__(self, training_info, validation_info, training_loss=None):
        if self.vl_ndata is None or self.tr_ndata is None or self.vl_suffix is None:
            raise ValueError(
                "FitState cannot be instantiated until vl_ndata, tr_ndata and vl_suffix are filled"
            )
        self._training = training_info
        self.validation = validation_info
        self._parsed = False
        self._vl_chi2 = None  # These are per replica
        self._tr_chi2 = None  # This is an overall training chi2
        self._vl_dict = None
        self._tr_dict = None
        # This can be given if ``training_info`` is not given
        self._training_loss = training_loss

    @property
    def vl_loss(self):
        """Return the total validation loss as it comes from the info dictionaries"""
        return self.validation.get("loss")

    @property
    def tr_loss(self):
        """Return the total validation loss as it comes from the info dictionaries"""
        if self._training is None:
            return self._training_loss
        return self._training.get("loss")

    def _parse_chi2(self):
        """
        Parses the chi2 from the losses according to the `tr_ndata` and
        `vl_ndata` dictionaries of {dataset: n_points}
        """
        if self._parsed:
            return
        if self._training is not None:
            self._tr_chi2, self._tr_dict = parse_losses(self._training, self.tr_ndata)
        if self.validation is not None:
            self._vl_chi2, self._vl_dict = parse_losses(
                self.validation, self.vl_ndata, suffix=self.vl_suffix
            )

    @property
    def tr_chi2(self):
        self._parse_chi2()
        return self._tr_chi2

    @property
    def vl_chi2(self):
        self._parse_chi2()
        return self._vl_chi2

    @property
    def all_tr_chi2(self):
        self._parse_chi2()
        return self._tr_dict

    @property
    def all_vl_chi2(self):
        self._parse_chi2()
        return self._vl_dict

    def all_tr_chi2_for_replica(self, i_replica):
        """Return the tr chi2 per dataset for a given replica"""
        return {k: np.take(v, i_replica) for k, v in self.all_tr_chi2.items()}

    def all_vl_chi2_for_replica(self, i_replica):
        """Return the vl chi2 per dataset for a given replica"""
        return {k: np.take(v, i_replica) for k, v in self.all_vl_chi2.items()}

    def total_partial_tr_chi2(self):
        """Return the tr chi2 summed over replicas per experiment"""
        return {k: np.sum(v) for k, v in self.all_tr_chi2.items()}

    def total_partial_vl_chi2(self):
        """Return the vl chi2 summed over replicas per experiment"""
        return {k: np.sum(v) for k, v in self.all_vl_chi2.items()}

    def total_tr_chi2(self):
        """Return the total tr chi2 summed over replicas"""
        return np.sum(self.tr_chi2)

    def total_vl_chi2(self):
        """Return the total vl chi2 summed over replicas"""
        return np.sum(self.vl_chi2)

    def __str__(self):
        return f"chi2: tr={self.tr_chi2} vl={self.vl_chi2}"


class FitHistory:
    """
    Keeps a list of FitState items holding the full chi2 history of the fit.

    Parameters
    ----------
        tr_ndata: dict
            dictionary of {dataset: n_points} for the training data
        vl_ndata: dict
            dictionary of {dataset: n_points} for the validation data
    """

    def __init__(self, tr_ndata, vl_ndata):
        if vl_ndata is None:
            vl_ndata = tr_ndata
            vl_suffix = "loss"
        else:
            vl_suffix = "val_loss"
        # All instances of FitState should use these
        FitState.tr_ndata = tr_ndata
        FitState.vl_ndata = vl_ndata
        FitState.vl_suffix = vl_suffix

        # Save a list of status for the entire fit
        self._history = []
        self.final_epoch = None

    def get_state(self, epoch):
        """Get the FitState of the system for a given epoch"""
        try:
            return self._history[epoch]
        except IndexError as e:
            raise ValueError(
                f"Tried to get obtain the state for epoch {epoch} when only {len(self._history)} epochs have been saved"
            ) from e

    def register(self, epoch, fitstate):
        """Save the current fitstate and the associated epoch
        and set the current epoch as the final one should the fit end now
        """
        self.final_epoch = epoch
        self._history.append(fitstate)


class Stopping:
    """
    Driver of the stopping algorithm

    Note, if the total number of points in the validation dictionary is None, it is assumed
    the validation_model actually corresponds to the training model.

    Parameters
    ----------
        validation_model: n3fit.backends.MetaModel
           the model with the validation mask applied
           (and compiled with the validation data and covmat)
        all_data_dicts: dict
           list containg all dictionaries containing all information about
           the experiments/validation/regularizers/etc to be parsed by Stopping
        pdf_model: n3fit.backends.MetaModel
           pdf_model being trained
        threshold_positivity: float
           maximum value allowed for the sum of all positivity losses
        total_epochs: int
           total number of epochs
        stopping_patience: int
           how many epochs to wait for the validation loss to improve
        threshold_chi2: float
            maximum value allowed for chi2
        dont_stop: bool
           dont care about early stopping
    """

    def __init__(
        self,
        validation_model,
        all_data_dicts,
        pdf_model,
        threshold_positivity=THRESHOLD_POS,
        total_epochs=0,
        stopping_patience=7000,
        threshold_chi2=10.0,
        dont_stop=False,
    ):
        self._pdf_model = pdf_model

        # Save the validation object
        self._validation = validation_model

        # Create the History object
        tr_ndata, vl_ndata, pos_sets = parse_ndata(all_data_dicts)
        self._history = FitHistory(tr_ndata, vl_ndata)

        # And the positivity checker
        self._positivity = Positivity(threshold_positivity, pos_sets)

        # Initialize internal variables for the stopping
        self._n_replicas = pdf_model.num_replicas
        self._threshold_chi2 = threshold_chi2
        self._stopping_degrees = np.zeros(self._n_replicas, dtype=int)
        self._counts = np.zeros(self._n_replicas, dtype=int)
        # Keep track of the replicas that should not be stopped yet
        self._dont_stop_me_now = np.ones(self._n_replicas, dtype=bool)

        self._dont_stop = dont_stop
        self._stop_now = False
        self.stopping_patience = stopping_patience
        self.total_epochs = total_epochs

        self._stop_epochs = [total_epochs - 1] * self._n_replicas
        self._best_epochs = [None] * self._n_replicas
        self.positivity_statuses = [POS_BAD] * self._n_replicas
        self._best_weights = [None] * self._n_replicas
        self._best_val_chi2s = [INITIAL_CHI2] * self._n_replicas

    @property
    def vl_chi2(self):
        """Current validation chi2"""
        validation_info = self._validation.compute_losses()
        fitstate = FitState(None, validation_info)
        return fitstate.vl_chi2

    @property
    def e_best_chi2(self):
        """Epoch of the best chi2, if there is no best epoch, return last"""
        best_or_last_epochs = [
            best if best is not None else last
            for best, last in zip(self._best_epochs, self._stop_epochs)
        ]
        return best_or_last_epochs

    @property
    def stop_epoch(self):
        """Epoch in which the fit is stopped"""
        return -1 if self._history.final_epoch is None else self._history.final_epoch + 1

    @property
    def positivity_status(self):
        """Returns POS_PASS if positivity passes or veto if it doesn't
        for each replica"""
        return self.positivity_statuses

    def evaluate_training(self, training_model):
        """Given the training model, evaluates the
        model and parses the chi2 of the training datasets

        Parameters
        ----------
            training_model: n3fit.backends.MetaModel
                an object implementing the evaluate function

        Returns
        -------
            tr_chi2: float
                chi2 of the given ``training_model``
        """
        training_info = training_model.compute_losses()
        fitstate = FitState(training_info, None)
        return fitstate.tr_chi2

    def monitor_chi2(self, training_info, epoch, print_stats=False):
        """
        Function to be called at the end of every epoch.
        Stores the total chi2 of the training set as well as the
        total chi2 of the validation set.
        If the training chi2 is below a certain threshold,
        stores the state of the model which gave the minimum chi2
        as well as the epoch in which occurred
        If the epoch is a multiple of save_all_each then we also save the per-exp chi2

        Returns True if the run seems ok and False if a NaN is found

        Parameters
        ----------
            training_info: dict
                output of a .fit() call, dictionary of the total training loss
                (summed over replicas and experiments)
            epoch: int
                index of the epoch

        Returns
        -------
            pass_ok: bool
                true/false according to the status of the run
        """
        # Step 1. Check whether the fit has NaN'd and stop it if so
        if np.isnan(training_loss := training_info["loss"]):
            log.warning(" > NaN found, stopping activated")
            self.make_stop()
            return False

        # Step 2. Compute the validation metrics
        validation_info = self._validation.compute_losses()

        # Step 3. Register the current point in (the) history
        # and set the current final epoch as the current one
        fitstate = FitState(None, validation_info, training_loss)
        self._history.register(epoch, fitstate)
        if print_stats:
            self.print_current_stats(epoch, fitstate)

        # Step 4. Check whether this is a better fit
        #         this means improving vl_chi2 and passing positivity
        # Don't start counting until the chi2 of the validation goes below a certain threshold
        # once we start counting, don't bother anymore
        passes = self._counts | (fitstate.vl_chi2 < self._threshold_chi2)
        passes &= fitstate.vl_loss < self._best_val_chi2s
        # And the ones that pass positivity
        passes &= self._positivity(fitstate)
        # Stop replicas that are ok being stopped (because they are finished or otherwise)
        passes &= self._dont_stop_me_now

        self._stopping_degrees += self._counts

        # Step 5. loop over the valid indices to check whether the vl improved
        for i_replica in np.where(passes)[0]:
            self._best_epochs[i_replica] = epoch
            # By definition, if we have a ``best_epoch`` then positivity passed
            self.positivity_statuses[i_replica] = POS_OK

            self._best_val_chi2s[i_replica] = self._history.get_state(epoch).vl_loss[i_replica]
            self._best_weights[i_replica] = self._pdf_model.get_replica_weights(i_replica)

            self._stopping_degrees[i_replica] = 0
            self._counts[i_replica] = 1

        stop_replicas = self._counts & (self._stopping_degrees > self.stopping_patience)
        for i_replica in np.where(stop_replicas)[0]:
            self._stop_epochs[i_replica] = epoch
            self._counts[i_replica] = 0
            self._dont_stop_me_now[i_replica] = False

        # By using the stopping degree we only stop when none of the replicas are improving anymore
        if min(self._stopping_degrees) > self.stopping_patience:
            self.make_stop()
        return True

    def make_stop(self):
        """Convenience method to set the stop_now flag
        and reload the history to the point of the best model if any
        """
        self._stop_now = True
        self._restore_best_weights()

    def _restore_best_weights(self):
        for i_replica, weights in enumerate(self._best_weights):
            if weights is not None:
                self._pdf_model.set_replica_weights(weights, i_replica)

    def print_current_stats(self, epoch, fitstate):
        """
        Prints ``fitstate`` validation chi2 for every experiment
        and the current total training loss as well as the validation loss
        after the training step
        """
        epoch_index = epoch + 1
        vl_chi2 = fitstate.total_vl_chi2()
        total_str = f"Epoch {epoch_index}/{self.total_epochs}: loss: {fitstate.tr_loss:.7f}"
        total_str += f"\nValidation loss after training step: {vl_chi2:.7f}."

        # The partial chi2 makes no sense for more than one replica at once:
        if self._n_replicas == 1:
            total_str += "\nValidation chi2s: "
            partial_vl_chi2 = fitstate.total_partial_vl_chi2()
            partials = []
            for experiment, chi2 in partial_vl_chi2.items():
                partials.append(f"{experiment}: {chi2:.3f}")
            total_str += ", ".join(partials)
        log.info(total_str)

    def stop_here(self):
        """Returns the stopping status
        If `dont_stop` is set returns always False (i.e., never stop)
        """
        if self._dont_stop:
            return False
        else:
            return self._stop_now

    def chi2exps_json(self, i_replica=0, log_each=100):
        """
        Returns and apt-for-json dictionary with the status of the fit every `log_each` epochs
        It reports the total training loss and the validation loss broken down by experiment.

        Parameters
        ----------
            i_replica: int
                which replica are we writing the log for
            log_each: int
                every how many epochs to print the log

        Returns
        -------
            file_list: list(str)
                a list of strings to be printed as `chi2exps.log`
        """
        final_epoch = self._history.final_epoch
        json_dict = {}

        for epoch in range(log_each - 1, final_epoch + 1, log_each):
            fitstate = self._history.get_state(epoch)
            # Get the training and validation losses
            tmp = {"training_loss": fitstate.tr_loss, "validation_loss": fitstate.vl_loss.tolist()}

            # And the validation chi2 broken down by experiment

            tmp["validation_chi2s"] = fitstate.all_vl_chi2_for_replica(i_replica)
            json_dict[epoch + 1] = tmp

        return json_dict


class Positivity:
    """
    Controls the positivity requirements.

    In order to check the positivity passes will check the history of the fitting
    as the fitting included positivity sets.
    If the sum of all positivity sets losses is above a certain value the model is
    not accepted and the training continues.

    Parameters
    ----------
        threshold_positivity: float
            maximum value allowed for the sum of all positivity losses
        positivity_sets: list
            list of positivity datasets
    """

    def __init__(self, threshold, positivity_sets):
        self.threshold = threshold
        self.positivity_sets = positivity_sets

    def check_positivity(self, history_object):
        """
                This function receives a history objects and loops over the
                positivity_sets to check the value of the positivity loss.

                If the positivity loss is above the threshold, the positivity fails
                otherwise, it passes.
                It returns an array booleans which are True if positivity passed
        story_object[key_loss] < self.threshold

                Parameters
                ----------
                    history_object: dict
                        dictionary of entries in the form  {'name': loss}, output of a MetaModel .fit()
        """
        positivity_pass = True
        for key in self.positivity_sets:
            key_loss = f"{key}_loss"
            positivity_pass &= history_object[key_loss] < self.threshold
        return np.array(positivity_pass)

    def __call__(self, fitstate):
        """
        Checks whether a given FitState object
        passes the positivity requirement
        """
        return self.check_positivity(fitstate.validation)
